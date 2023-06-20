import numpy as np
from gym import spaces
import torch
from segment_tree import SumSegmentTree, MinSegmentTree
import random

'''
Class definitions for replay buffer used in off-policy training
'''

class ContinuousSequenceReplayBuffer:
    def __init__(self, buffer_size, observation_space,
                 action_space, hidden_state_size, sequence_length=1,
                 burn_in_length=0, n_envs=1,
                 alpha=0.6, beta=0.4, 
                 beta_increment=0.0001, max_priority=1.0):
        '''
        A replay buffer for R2D2 algorithm that when sampled, produces sequences of time steps.
        Any index can be samples from, and burn_in_length steps before the index and sequence_length
          steps after the index will be passed together
        Note that there is one torch tensor per variable (observations, actions, rewards, dones, rnn_hxs)
          and it will continuously be overwritten. Each tensor has length burn_in_length+buffer_size+sequence_length.
          Think of it as having buffer_size, plus a chunk behind and ahead to handle burn in and sequence.
          
        self.pos keeps track of the next index to be written to. When it reaches the end (burn_in_length+buffer_size)
          it loops back to the start (burn_in_length).
          When it loops back, the burn_in_length chunk is copied from the end of the buffer.
          As it covers the the first sequence_length worth of steps in the buffer, these get copied to the end
            sequence_length chunk of the buffer
        
        buffer_size: number of steps to hold in buffer
        sequence_length: number of steps in sequence
        burn_in_length: number of steps before idx to be passed with sequence
        
        Priorities
        '''
        self.buffer_size = buffer_size
        total_buffer_size = buffer_size + sequence_length + burn_in_length
        self.n_envs = n_envs
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = max_priority

        self.td_priorities = np.zeros(total_buffer_size*n_envs) #holds individual td errors for priority calculations
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < total_buffer_size*n_envs:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity) #trees hold actual priorities for faster updating and sampling
        self.min_tree = MinSegmentTree(tree_capacity)
        self.total_buffer_size = total_buffer_size
        
        action_shape = get_action_dim(action_space)
        self.observations = np.zeros((total_buffer_size, n_envs, *observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((total_buffer_size, n_envs, action_shape), dtype=action_space.dtype)
        self.rewards = np.zeros((total_buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((total_buffer_size, n_envs), dtype=np.float32)
        self.hidden_states = np.zeros((total_buffer_size, n_envs, hidden_state_size), dtype=np.float32)

        
        self.pos = burn_in_length
        self.full = False
        
    def add(self, obs, next_obs, action, reward, done, hidden_state):
        '''
        Add to the buffer
        '''
        bil = self.burn_in_length
        bs = self.buffer_size
        sl = self.sequence_length

        
        self.observations[self.pos] = np.array(obs).copy()
        self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.hidden_states[self.pos] = np.array(hidden_state).copy()
        
        for i in range(self.n_envs):
            self.td_priorities[self.pos + i*self.total_buffer_size] = self.max_priority
            # 0 out probabilities for indexes that become invalid
            self.sum_tree[self.pos + i*self.total_buffer_size] = 0.
            self.sum_tree[self.pos + i*self.total_buffer_size + bil] = 0.
        if (self.full and self.pos >= bil + sl) or (self.pos >= 2*bil + sl):
            # only update steps in the past 
            #  there is a very slight chance this overwrites some priority that was set in 
            #  an update step but shouldn't make a huge difference
            for i in range(self.n_envs):
                self.sum_tree[self.pos + i*self.total_buffer_size - sl] = self.max_priority ** self.alpha
                self.min_tree[self.pos + i*self.total_buffer_size - sl] = self.max_priority ** self.alpha
        
        
        #Make copies to extra end portion
        #Note that this makes it so that for a sequence_length period of time
        #while we are filling up the end of the buffer, end steps cannot be used
        if self.pos < bil + sl:
            self.observations[self.pos+bs] = self.observations[self.pos].copy()
            self.actions[self.pos+bs] = self.actions[self.pos].copy()
            self.rewards[self.pos+bs] = self.rewards[self.pos].copy()
            self.dones[self.pos+bs] = self.dones[self.pos].copy()
            self.hidden_states[self.pos+bs] = self.hidden_states[self.pos].copy()
            
            # Copies of the end need to be made for td_priorities, but burn-in does not need
            #  These priorities are not true priorities, just used to later calculate total td
            #  for sequences. Hence they do not get copied to the sum_tree or min_tree
            for i in range(self.n_envs):
                self.td_priorities[self.pos+bs + i*self.total_buffer_size] = self.max_priority
                # self.sum_tree[self.pos+bs + i*self.total_buffer_size] = self.max_priority ** self.alpha
                # self.min_tree[self.pos+bs + i*self.total_buffer_size] = self.max_priority ** self.alpha

            
        self.pos += 1
        if self.pos == self.buffer_size + self.burn_in_length:
            self.pos = self.burn_in_length
            self.full = True
            
            #Make copies to the burn_in portion
            self.observations[:bil] = self.observations[bs:bs+bil].copy()
            self.actions[:bil] = self.actions[bs:bs+bil].copy()
            self.rewards[:bil] = self.rewards[bs:bs+bil].copy()
            self.dones[:bil] = self.dones[bs:bs+bil].copy()
            self.hidden_states[:bil] = self.hidden_states[bs:bs+bil].copy()

            
    def _sample_indices(self, num_sequences):
        '''
        Use sum tree to sample indices from priorities in segments
        '''
        t_indices = []
        env_indices = []
        p_total = self.sum_tree.sum()
        segment = p_total / num_sequences
        
        # Check if stratified sampling will be valid based on number
        #  of sequences asked for and fullness of storage        
        for i in range(num_sequences):
            found = False
            sample_attempt_count = 0
            valid_idxs = self.get_valid_idxs()
            
            a = segment * i
            b = segment * (i + 1)
            
            while not found and sample_attempt_count < 50:
                upperbound = random.uniform(a, b)
                idx = self.sum_tree.retrieve(upperbound)
                # print(idx)
                t_index = idx % self.total_buffer_size
                if valid_idxs[t_index]:
                    found = True
                sample_attempt_count += 1
            
            if sample_attempt_count >= 50:
                # If this happens, either buffer is not being filled enough before
                #  samples are being called for, or there is a bug
                print('Warning: sample index failed to find valid index 50 times')
                
            t_indices.append(idx % self.total_buffer_size)
            env_indices.append(idx // self.total_buffer_size)
            
        
        return np.array(t_indices), np.array(env_indices)
        
        
    def _calculate_weight(self, t_idx, env_idx):
        '''Calculate the weight of the experience at idx.'''
        # print(t_idx, env_idx)
        
        size = self.buffer_size if self.full else self.pos
        size = size * self.n_envs
        idx = t_idx + env_idx*self.total_buffer_size
        
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * size) ** (-self.beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * size) ** (-self.beta)
        weight = weight / max_weight
        
        return weight
        
    def sample(self, num_sequences):
        '''
        Generate a sample of data to be trained with from the buffer
        Note that we will actually generate a total training batch size of
            num_sequences*self.sequence_length
        and a total number of steps returned of
            num_sequences*(self.sequence_length+self.burn_in_length).
        It is up to the one calling sample to take into account how many
        sequence samples it wants
        '''
        t_idxs, env_idxs = self._sample_indices(num_sequences)
        start_idxs = t_idxs - self.burn_in_length
        
        window_idxs = np.arange(-self.burn_in_length, self.sequence_length)
        window_length = len(window_idxs)
        seq_idxs = t_idxs[:, np.newaxis] + window_idxs
        seq_env_idxs = np.full((num_sequences, window_length), env_idxs[:, np.newaxis])
        
        # weights are [N, 1] tensor to be multiplied to each sequence batch generaated
        weights = torch.Tensor([self._calculate_weight(t_idxs[i], env_idxs[i]) \
                                for i in range(num_sequences)]).reshape(-1, 1)

        self.beta = min(1.0, self.beta + self.beta_increment)
                
        obs = torch.Tensor(self.observations[seq_idxs, seq_env_idxs])
        next_obs = torch.Tensor(self.observations[seq_idxs+1, seq_env_idxs])
        actions = torch.Tensor(self.actions[seq_idxs, seq_env_idxs])
        rewards = torch.Tensor(self.rewards[seq_idxs, seq_env_idxs])
        dones = torch.Tensor(self.dones[seq_idxs, seq_env_idxs])
        next_dones = torch.Tensor(self.dones[seq_idxs+1, seq_env_idxs])
        
        hidden_states = torch.Tensor(self.hidden_states[start_idxs, env_idxs]).unsqueeze(0)
        next_hidden_states = torch.Tensor(self.hidden_states[start_idxs+1, env_idxs]).unsqueeze(0)
        
        sample = {
            'observations': obs,
            'next_observations': next_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_dones': next_dones,
            'hidden_states': hidden_states,
            'next_hidden_states': next_hidden_states,
            'weights': weights,
            't_idxs': t_idxs,
            'seq_idxs': seq_idxs,
            'env_idxs': env_idxs,
        }
        
        return sample
    
    
    def get_valid_idxs(self):
        '''
        Get array of valid indexes that can be sampled. Valid indexes are those with enough time steps
        of data earlier to match burn_in_lenth and with enough time steps of data later to match
        sequence_length.
        
        return a self.total_buffer_size (bil+buffer_size+seq_len) boolean array
            that can be checked for truth by indexing directly
        
        Note: there is a slight bug here to be fixed in the future - there is a period where self.pos
          loops back that the sequence_length post buffer is stale until overwritten fully 
        '''        
        start = self.burn_in_length
        end = self.burn_in_length + self.buffer_size
        valid_idxs = np.full(self.total_buffer_size, False)
        if self.full:
            #Have enough terms ahead to be usable
            valid_idxs[start:self.pos - self.sequence_length] = True
            #Have enough terms behind to be usable
            valid_idxs[self.pos + self.burn_in_length-1:end] = True
        else:
            #First burn_in_length steps are not valid because they haven't been copied
            valid_idxs[start + self.burn_in_length:self.pos - self.sequence_length] = True
        
        return valid_idxs

    def update_priorities(self, seq_idxs, env_idxs, priorities):
        '''
        seq_idxs: shape [N, seq_len]
        env_idxs: shape [N,] for N batches
        priorities: shape [N, seq_len]
        '''
        
        valid_idxs = self.get_valid_idxs()
        
        n_batches = len(env_idxs)
        for i in range(n_batches):
            update_priority_idxs = seq_idxs[i] + env_idxs[i] * self.total_buffer_size
            self.td_priorities[update_priority_idxs] = priorities[i]

            for j in range(len(update_priority_idxs)):
                start = seq_idxs[i, j] + env_idxs[i]*self.total_buffer_size

                # check if the updated priority needs to be copied to buffer end segment
                if seq_idxs[i, j] < self.burn_in_length + self.sequence_length:
                    copy_idx = start + self.buffer_size
                    self.td_priorities[copy_idx] = priorities[i, j]

                if valid_idxs[seq_idxs[i, j]]:
                    # next update priority based on future td_steps
                    avg_td_priority = self.td_priorities[start:start+self.sequence_length].sum() / self.sequence_length
                    self.sum_tree[start] = avg_td_priority ** self.alpha
                    self.min_tree[start] = avg_td_priority ** self.alpha

    def __len__(self):
        return len(self.buffer)
    
    
    
    
class SequenceReplayBuffer:
    def __init__(self, buffer_size, observation_space,
                 action_space, hidden_state_size, sequence_length=8,
                 burn_in_length=4, n_envs=1,
                 alpha=0.6, beta=0.4, 
                 beta_increment=0.0001, max_priority=1.0):
        '''
        A replay buffer for R2D2 algorithm that when sampled, produces sequences of time steps.
        Will continually store steps until a done is received or max sequence length is reached
            then store that sequence into the replay buffer
          
        self.pos keeps track of the next index to be written to. When it reaches the end 
          it loops back to the start.
        
        buffer_size: number of sequences to hold in buffer
        sequence_length: number of steps in sequence
        burn_in_length: number of steps before idx to be passed with sequence
        '''
        self.buffer_size = buffer_size
        self.bil_sl = burn_in_length + sequence_length
        total_buffer_size = buffer_size + sequence_length + burn_in_length
        self.n_envs = n_envs
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = max_priority

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity) #trees hold actual priorities for faster updating and sampling
        self.min_tree = MinSegmentTree(tree_capacity)
        
        # buffer shape [buffer_size, sequence_length, data_dim]
        #  note that we add to the buffer regardless of which env the sequence comes from
        
        action_shape = get_action_dim(action_space)
        self.observations = np.zeros((buffer_size, self.bil_sl, *observation_space.shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((buffer_size, self.bil_sl, *observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size, self.bil_sl, action_shape), dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size, self.bil_sl), dtype=np.float32)
        self.dones = np.zeros((buffer_size, self.bil_sl), dtype=np.float32)
        self.hidden_states = np.zeros((buffer_size, self.bil_sl, hidden_state_size), dtype=np.float32)
        self.next_hidden_states = np.zeros((buffer_size, self.bil_sl, hidden_state_size), dtype=np.float32)
        # training_masks is used to keep track of which steps are trainable
        #  note that it only has length sequence_length, as opposed to bil+sl
        self.training_masks = np.zeros((buffer_size, self.sequence_length), dtype=np.float32)

        self.cur_observations = np.zeros((n_envs, self.bil_sl, *observation_space.shape), dtype=observation_space.dtype)
        self.cur_next_observations = np.zeros((n_envs, self.bil_sl, *observation_space.shape), dtype=observation_space.dtype)
        self.cur_actions = np.zeros((n_envs, self.bil_sl, action_shape), dtype=action_space.dtype)
        self.cur_rewards = np.zeros((n_envs, self.bil_sl), dtype=np.float32)
        self.cur_dones = np.zeros((n_envs, self.bil_sl), dtype=np.float32)
        self.cur_hidden_states = np.zeros((n_envs, self.bil_sl, hidden_state_size), dtype=np.float32)
        self.cur_next_hidden_states = np.zeros((n_envs, self.bil_sl, hidden_state_size), dtype=np.float32)

        self.pos = 0
        
        self.cur_pos = np.zeros(n_envs, dtype='long') # keep track of which environments are done
        self.full = False
        
        
    def add(self, obs, next_obs, action, reward, done, hidden_state, next_hidden_state):
        '''
        Add to the buffer. Each incoming input should be of shape
            [n_envs, data_dim]
        '''
        for i in range(self.n_envs):
            self.cur_observations[i, self.cur_pos[i]] = np.array(obs[i]).copy()
            self.cur_next_observations[i, self.cur_pos[i]] = np.array(next_obs[i]).copy()
            self.cur_actions[i, self.cur_pos[i]] = np.array(action[i]).copy()
            self.cur_rewards[i, self.cur_pos[i]] = np.array(reward[i]).copy()
            self.cur_dones[i, self.cur_pos[i]] = np.array(done[i]).copy()
            self.cur_hidden_states[i, self.cur_pos[i]] = np.array(hidden_state[:, i, :]).copy()
            self.cur_next_hidden_states[i, self.cur_pos[i]] = np.array(next_hidden_state[:, i, :]).copy()
            self.cur_pos[i] += 1

            if done[i] or self.cur_pos[i] == self.bil_sl:
                # copy the sequence to the buffer
                self.observations[self.pos] = self.cur_observations[i]
                self.next_observations[self.pos] = self.cur_next_observations[i]
                self.actions[self.pos] = self.cur_actions[i]
                self.rewards[self.pos] = self.cur_rewards[i]
                self.dones[self.pos] = self.cur_dones[i]
                self.hidden_states[self.pos] = self.cur_hidden_states[i]
                self.next_hidden_states[self.pos] = self.cur_next_hidden_states[i]

                trainable_steps = self.cur_pos[i] - self.burn_in_length
                training_mask = np.zeros((self.sequence_length,))
                training_mask[:trainable_steps] = 1
                self.training_masks[self.pos] = training_mask
                                
                # make copy of the last steps to carry over to next sequence
                copy_steps = min(self.cur_pos[i], self.burn_in_length)
                
                # only copy if not done. If done, start a new sequence up
                if not done[i]:
                    self.cur_observations[i, :copy_steps] = self.cur_observations[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                    self.cur_next_observations[i, :copy_steps] = self.cur_next_observations[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                    self.cur_actions[i, :copy_steps] = self.cur_actions[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                    self.cur_rewards[i, :copy_steps] = self.cur_rewards[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                    self.cur_dones[i, :copy_steps] = self.cur_dones[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                    self.cur_hidden_states[i, :copy_steps] = self.cur_hidden_states[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                    self.cur_next_hidden_states[i, :copy_steps] = self.cur_next_hidden_states[i, self.cur_pos[i]-copy_steps:self.cur_pos[i]]
                else:
                    copy_steps = 0
                
                self.cur_observations[i, copy_steps:] = 0.
                self.cur_next_observations[i, copy_steps:] = 0.
                self.cur_actions[i, copy_steps:] = 0.
                self.cur_rewards[i, copy_steps:] = 0.
                self.cur_dones[i, copy_steps:] = 0.
                self.cur_hidden_states[i, copy_steps:] = 0.
                self.cur_next_hidden_states[i, copy_steps:] = 0.
                
                self.cur_pos[i] = copy_steps
                
                self.sum_tree[self.pos] = self.max_priority ** self.alpha
                self.min_tree[self.pos] = self.max_priority ** self.alpha
                
                self.pos = (self.pos + 1) % self.buffer_size

                
            
    def _sample_proportional(self, num_sequences):
        '''
        Use sum tree to sample indices from priorities in segments
        '''
        indices = []
        p_total = self.sum_tree.sum()
        segment = p_total / num_sequences
        
        # Check if stratified sampling will be valid based on number
        #  of sequences asked for and fullness of storage        
        for i in range(num_sequences):
            a = segment * i
            b = segment * (i + 1)
            
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
                    
        return indices
    
    
    def _sample_uniform(self, num_sequences=1):
        '''
        Use sum tree to get n sample indices without stratified sampling
        '''
        indices = []
        p_total = self.sum_tree.sum()
        total_trainable_steps = 0

        for i in range(num_sequences):
            upperbound = random.uniform(0, p_total)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            total_trainable_steps += (self.training_masks[idx].sum())

        return indices, total_trainable_steps
        
        
    def _calculate_weight(self, idx):
        '''Calculate the weight of the experience at idx.'''
        # print(t_idx, env_idx)
        
        size = self.buffer_size if self.full else self.pos
        size = size * self.n_envs
        
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * size) ** (-self.beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * size) ** (-self.beta)
        weight = weight / max_weight
        
        return weight
        
        
    def sample(self, num_steps=None, num_sequences=None):
        '''
        Generate a sample, either by number of sequences or number of steps
        Given a number of sequences, use stratified sampling
        '''
        if num_sequences is not None:
            idxs = self._sample_indices(num_sequences)
        elif num_steps is not None:
            idxs = []
            num_fails = 0
            trainable_steps_batched = 0
            while trainable_steps_batched < num_steps and num_fails < 50:
                new_idxs, n_trainable_steps = self._sample_uniform(1)
                idxs += new_idxs
                if n_trainable_steps <= 0:
                    num_fails += 1
                else:
                    trainable_steps_batched += n_trainable_steps
            
            if num_fails >= 50:
                print('Warning - sampling failed 50 times')
        else:
            raise Exception('One of num_steps or num_sequences must be given')
            
        
        # weights are [N, 1] tensor to be multiplied to each sequence batch generaated
        weights = torch.Tensor([self._calculate_weight(idxs[i]) \
                                for i in range(len(idxs))]).reshape(-1, 1)

        self.beta = min(1.0, self.beta + self.beta_increment)
                
        obs = torch.Tensor(self.observations[idxs])
        next_obs = torch.Tensor(self.next_observations[idxs])
        actions = torch.Tensor(self.actions[idxs])
        rewards = torch.Tensor(self.rewards[idxs])
        dones = torch.Tensor(self.dones[idxs])
        next_dones = torch.Tensor(self.dones[idxs])
        hidden_states = torch.Tensor(self.hidden_states[idxs, 0, :]).unsqueeze(0)
        next_hidden_states = torch.Tensor(self.next_hidden_states[idxs, 0, :]).unsqueeze(0)
        training_masks = torch.Tensor(self.training_masks[idxs])
        
        sample = {
            'observations': obs,
            'next_observations': next_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_dones': next_dones,
            'hidden_states': hidden_states,
            'next_hidden_states': next_hidden_states,
            'training_masks': training_masks,
            'weights': weights,
            'idxs': idxs,
        }
        
        return sample
    

    def update_priorities(self, idxs, priorities):
        '''
        idxs: shape [N,]
        priorities: shape [N,]
        '''
        
        assert len(idxs) == len(priorities)

        for idx, priority in zip(idxs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)


    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.pos
    
    
    
def get_action_dim(action_space):
    """
    Get the dimension of the action space.
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
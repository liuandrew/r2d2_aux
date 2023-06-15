import torch
from torch import nn
import torch.nn.functional as F
from model import RNNQNetwork, linear_schedule
from storage import SequenceReplayBuffer
from envs import make_vec_envs
import torch.optim as optim
import random
import numpy as np
import gym
import gym_nav

class R2D2Agent(nn.Module):
    def __init__(self, batch_size=128, burn_in_length=4, sequence_length=8,
                 gamma=0.99, tau=1., learning_rate=2.5e-4, hidden_size=64,
                 device=torch.device('cpu'), buffer_size=10_000, 
                 learning_starts=10_000, train_frequency=10, target_network_frequency=500,
                 total_timesteps=30_000, start_e=1., end_e=0.05, exploration_fraction=0.5, 
                 seed=None, num_envs=1, dummy_env=True,
                 env_id='CartPole-v1', env_kwargs={},
                 verbose=0, q_network=None,  deterministic=False, env=None
                 ):
        """
        R2D2 setup following same parameters as args.py has
        verbose: Level of verbosity of print statements
            1: print episode lengths and returns
        q_network: Mostly for use of evaluation with a saved q_network
          optionally pass in a q_network to use manually
        deterministic: If True, manually set epsilon to 0 for every act() call
        env: Also option to manually pass in an environment
        num_envs: option to make multiple envs and have q_network generate multiple
        dummy_env: whether to use DummyVecEnv as opposed to SubprocVecEnv for testing
        """
        
        super().__init__()
        
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.device=device

        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction

        self.burn_in_length = burn_in_length
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_envs = num_envs

        self.seed = seed
        self.deterministic = deterministic
        if env == None:
            # self.env = gym.make(env_id, **env_kwargs)
            self.env = make_vec_envs(env_id, num_envs, env_kwargs=env_kwargs,
                                     dummy=dummy_env)
        else:
            self.env = env
        
        
        if q_network == None:
            self.q_network = RNNQNetwork(self.env, hidden_size).to(device)
        else:
            self.q_network = q_network
        self.target_network = RNNQNetwork(self.env, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.rb = SequenceReplayBuffer(buffer_size, self.env.observation_space, self.env.action_space,
                                hidden_size, sequence_length, burn_in_length, num_envs)

        
        self.global_step = 0
        self.global_update_step = 0
        self.rnn_hxs = self.q_network.get_rnn_hxs(self.num_envs)
        self.obs = self.env.reset()
        self.masks = torch.zeros((self.num_envs, 1), dtype=torch.float32)
        
        self.cur_episode_t = np.zeros(self.num_envs)
        self.cur_episode_r = np.zeros(self.num_envs)
        
        self.verbose = verbose
        
    
    def act(self, obs, rnn_hxs, epsilon=True, masks=None):
        """Compute q values and sample policy. If epsilon is True,
        perform randomo action with probability based on current global timestep
        
        masks: tensor of shape (N, 1) which has entries 0.0 when done
            and 1.0 when not done, indicating when rnn_hxs should be reset
            Used for vectorized environments
        """
        if epsilon:
            epsilon = linear_schedule(self.start_e, self.end_e, 
                        self.exploration_fraction*self.total_timesteps,
                        self.global_step)
        else:
            epsilon = 0

        
        obs_tensor = torch.Tensor(obs).to(self.device)
        if obs_tensor.dim() < rnn_hxs.dim():
            # We have an observation from the environment but need to unsqueeze
            #  to tell the GRU that this is an observation of time length 1
            # If it is batched (dim == 2), then we add an axis in the middle
            #  otherwise add it to the start
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                action_shape = (1,)
                action_dim = 1
            elif obs_tensor.dim() == 2:
                obs_tensor = obs_tensor.unsqueeze(1)
                action_shape = (obs_tensor.shape[0], 1)
                action_dim = 2
        
        else:
            if obs_tensor.dim() == 2:
                action_shape = (1, obs_tensor.shape[1],)
                action_dim = 1
            elif obs_tensor.dim() == 3:
                action_shape = (obs_tensor.shape[0], obs_tensor.shape[1],)
                action_dim = 2  
            
        q_values, gru_out, next_rnn_hxs = self.q_network(obs_tensor, rnn_hxs, masks=masks)
        
        if random.random() < epsilon:
            action = np.zeros(action_shape)
            if len(action_shape) == 2:
                for i in range(action_shape[0]):
                    for j in range(action_shape[1]):
                        action[i, j] = self.env.action_space.sample()
            else:
                for i in range(action_shape[0]):
                    action[i] = self.env.action_space.sample()
        else:
            # action = np.array([[q_values.argmax()]])
            action = q_values.argmax(dim=action_dim).numpy()

        if len(action.shape) == 1:
            action = action[np.newaxis, :]

        return action, q_values, next_rnn_hxs
                
        
    def collect(self, num_steps):
        """Perform policy for n steps and add to memory buffer
        
        Note that we will add a total of num_steps * self.num_envs to the buffer"""
        env = self.env
        
        for t in range(num_steps):
            action, q_values, next_rnn_hxs = self.act(self.obs, self.rnn_hxs, masks=self.masks)
            next_obs, reward, done, info = env.step(action)
            
            self.cur_episode_r += reward
            self.cur_episode_t += 1
            
            # if done:
            #     next_obs = env.reset()
            #     next_rnn_hxs = self.q_network.get_rnn_hxs()
                
            #     if self.verbose >= 1:
            #         print(f'Episode R: {self.cur_episode_r}, L: {self.cur_episode_t}')
                
            #     self.cur_episode_t = 0
            #     self.cur_episode_r = 0

            # Masks are used to reset hidden state when vectorized environmnts give dones
            self.masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            for i, done_ in enumerate(done):
                if done_:
                    if self.verbose >= 1:
                        print(f'Episode R: {self.cur_episode_r[i]}, L: {self.cur_episode_t[i]}')
                    self.cur_episode_r[i] = 0
                    self.cur_episode_t[i] = 0
                    

            self.rb.add(self.obs, next_obs, action, reward, done, self.rnn_hxs.detach())
            
            self.obs = next_obs
            self.rnn_hxs = next_rnn_hxs
            
            
            
    
    def update(self):
        """Sample from buffer and perform Q-learning"""
    
        sample = self.rb.sample(self.batch_size//self.sequence_length)
        states = sample['observations']
        next_states = sample['next_observations']
        hidden_states = sample['hidden_states']
        next_hidden_states = sample['next_hidden_states']
        actions = sample['actions']
        rewards = sample['rewards']
        dones = sample['dones']
        next_dones = sample['next_dones']
        
        with torch.no_grad():
            target_q, _ = self.target_network(next_states, next_hidden_states, next_dones)
            target_max, _ = target_q.max(dim=2)
            td_target = rewards + self.gamma * target_max * (1 - dones)
        old_q, _ = self.q_network(states, hidden_states, dones)
        old_val = old_q.gather(2, actions.long()).squeeze()

        loss = F.mse_loss(td_target[:, self.burn_in_length:], old_val[:, self.burn_in_length:])
                
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.global_update_step += 1
    

    def train(self, n_updates):
        if self.global_step < self.learning_starts:
            self.collect(self.learning_starts - self.global_step)
        
        for i in range(n_updates):
            self.collect(self.train_frequency)
            self.update()


    def get_rnn_hxs(self):
        return self.q_network.get_rnn_hxs(self.num_envs)
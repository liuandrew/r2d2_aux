import numpy as np
import torch


def evaluate(agent, seed=None, device=torch.device('cpu'), ret_info=1, 
             data_callback=None, num_episodes=10, verbose=0, 
             with_activations=False, deterministic=True,
             normalize=True, aux_wrapper_kwargs={}, auxiliary_truth_sizes=[]):
    '''
    agent: R2D2Agent with an act and env attached
    
    ret_info: level of info that should be tracked and returned
    capture_video: whether video should be captured for episodes
    env_kwargs: any kwargs to create environment with
    data_callback: a function that should be called at each step to pull information
        from the environment if needed. The function will take arguments
            def callback(actor_critic, rnn_hxs, obs, action, reward, done, data, stack=False):
        actor_critic: the actor_critic network
        recurrent_hidden_states: these are given in all data, but may want to use in computation
        obs: observation this step (after taking action) - 
            note that initial observation is never seen by data_callback
            also note that this observation will have the mean normalized
            so may instead want to call vec_envs.get_method('get_observation')
        action: actions this step
        reward: reward this step
        data: a data dictionary that will continuously be passed to be updated each step
            it will start as an empty dicionary, so keys must be initialized
        stack: a flag to say whether to stack and reset episodic observations.
            Doesn't have to be implemented, but since this evaluation function is saving all data
            in episodic format it will help to keep everything consistent
        first: a flag to indicate that the function is being passed the first step in the episode
            the function can choose whether to use information from the first step or not
        see below at nav_data_callback in this file for an example
    '''

    all_obs = []
    all_actions = []
    all_rewards = []
    all_rnn_hxs = []
    all_dones = []
    all_activations = []
    all_qs = []
    all_actor_features = []
    all_auxiliary_preds = []
    all_auxiliary_truths = []
    data = {}

    env = agent.env
    
    obs = env.reset()
    rnn_hxs = agent.get_rnn_hxs()

    
    use_epsilon = False if deterministic else True
    
    ep_obs = []
    ep_actions = []
    ep_rewards = []
    ep_rnn_hxs = []
    ep_dones = []
    ep_qs = []
    
    ep_auxiliary_preds = []
    ep_activations = []
    ep_auxiliary_truths = []
    
    for i in range(num_episodes):
        step = 0
        
        while True:
            # These are appended before since they are defined as previous step compared to other 
            #  information like actions and rewards
            ep_obs.append(obs)
            ep_rnn_hxs.append(rnn_hxs)
            if data_callback is not None and step == 0:
                data = data_callback(agent, rnn_hxs,
                    obs, [], [], [False], data, first=True)
            
            
            with torch.no_grad():
                outputs = agent.act(obs, rnn_hxs, use_epsilon=use_epsilon,)
                                        #    with_activations=with_activations)
                action = outputs['action']
                rnn_hxs = outputs['next_rnn_hxs']
                
            # Obser reward and next obs
            obs, rewards, dones, infos = env.step(action)
            
            ep_actions.append(action)
            ep_rewards.append(rewards)
            ep_dones.append(dones)
            ep_qs.append(outputs['q_values'])
            # all_actor_features.append(outputs['actor_features'])
            
            if 'auxiliary_preds' in outputs:
                ep_auxiliary_preds.append(outputs['auxiliary_preds'])
            
            if with_activations:
                ep_activations.append(outputs['activations'])

            if data_callback is not None:
                data = data_callback(agent, rnn_hxs,
                    obs, action, rewards, dones, data)
            else:
                data = {}
                
            if len(auxiliary_truth_sizes) > 0:
                auxiliary_truths = [[] for i in range(len(agent.auxiliary_output_sizes))]
                for info in infos:
                    if 'auxiliary' in info and len(info['auxiliary']) > 0:
                        for i, aux in enumerate(info['auxiliary']):
                            auxiliary_truths[i].append(aux)
                if len(auxiliary_truths) > 0:
                    auxiliary_truths = [torch.tensor(np.vstack(aux)) for aux in auxiliary_truths]
            
                ep_auxiliary_truths.append(auxiliary_truths)
            
            step += 1
            
            # Note that we are assuming single vectorized environment for the agents environment
            if dones[0]:
                all_obs.append(np.vstack(ep_obs))
                all_actions.append(np.vstack(ep_actions))
                all_rewards.append(np.vstack(ep_rewards))
                all_rnn_hxs.append(np.vstack(ep_rnn_hxs))
                all_dones.append(np.vstack(ep_dones))
                all_qs.append(np.vstack(ep_qs))
                
                all_auxiliary_preds.append(ep_auxiliary_preds)
                all_activations.append(ep_activations)
                all_auxiliary_truths.append(ep_auxiliary_truths)
                
                if data_callback is not None:
                    data = data_callback(agent, rnn_hxs,
                        obs, action, rewards, dones, data, stack=True)

                if verbose >= 2:
                    print(f'ep {i}, rew {np.sum(ep_rewards)}' )
                
                ep_obs = []
                ep_actions = []
                ep_rewards = []
                ep_rnn_hxs = []
                ep_dones = []
                ep_qs = []
                
                ep_auxiliary_preds = []
                ep_activations = []
                ep_auxiliary_truths = []
                
                break
                        

    if verbose >= 1:
        lens = [len(o) for o in all_obs]
        rews = [np.sum(r) for r in all_rewards]
        print(f'Mean reward: {np.mean(rews)}. Mean length: {np.mean(lens)}')

    return {
        'obs': all_obs,
        'actions': all_actions,
        'rewards': all_rewards,
        'hidden_states': all_rnn_hxs,
        'dones': all_dones,
        'data': data,
        'activations': all_activations,
        'qs': all_qs,
        'actor_features': all_actor_features,
        'auxiliary_preds': all_auxiliary_preds,
        'auxiliary_truths': all_auxiliary_truths,
    }



def nav_data_callback(agent, rnn_hxs, obs, action, reward, done, data, stack=False,
                      first=False):
    '''
    Add navigation data pos and angle to data object
    If stack is True, this function will handle stacking the data properly
    '''
    env = agent.env
    
    if 'pos' not in data:
        data['pos'] = []
    if 'angle' not in data:
        data['angle'] = []
    if 'ep_pos' not in data:
        data['ep_pos'] = []
    if 'ep_angle' not in data:
        data['ep_angle'] = []

    if stack:
        data['pos'].append(np.vstack(data['ep_pos']))
        data['angle'].append(np.vstack(data['ep_angle']))
        
        data['ep_pos'] = []
        data['ep_angle'] = []        
    elif not done[0]:
        pos = env.get_attr('character')[0].pos.copy()
        angle = env.get_attr('character')[0].angle
        data['ep_pos'].append(pos)
        data['ep_angle'].append(angle)
    
    return data
        
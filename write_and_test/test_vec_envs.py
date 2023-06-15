from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
import proplot
import torch
import gym
import gym_nav
import numpy as np
# %run ../r2d2_algo/r2d2_class.py

def make_env(env_id, seed, rank, capture_video=False,
                env_kwargs=None):
    def _thunk():
        if env_kwargs is not None:
            env = gym.make(env_id, **env_kwargs)
        else:
            env = gym.make(env_id)

        #Andy: add capture video wrapper
        if capture_video is not False and capture_video != 0 and rank == 0:
            # env = gym.wrappers.Monitor(env, './video', 
            #     video_callable=lambda t:t%capture_video==0, force=True)
            env = gym.wrappers.RecordVideo(env, './video',
                episode_trigger=lambda t:t%capture_video==0)

        env.seed(seed + rank)
        obs_shape = env.observation_space.shape

        return env
    return _thunk

#Andy: add capture video param, add optional env kwargs
def make_vec_envs(env_name,
                  num_processes=1,
                  seed=None,
                  device=torch.device('cpu'),
                  num_frame_stack=None,
                  capture_video=False,
                  normalize=False,
                  env_kwargs={},
                  auxiliary_tasks=[],
                  auxiliary_task_args=[]):
    
    if seed == None:
        seed = np.random.randint(0, 1e9)
    envs = [
        make_env(env_name, seed, i, capture_video,
                env_kwargs)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1 and normalize:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    # envs = VecPyTorch(envs, device)
    # envs = AuxVecPyTorch(envs, device, auxiliary_tasks, auxiliary_task_args)

    return envs

if __name__ == '__main__':
    n_envs = 32
    
    envs = make_vec_envs('NavEnv-v0', n_envs)
    
    import time
    run_times = []
    
    for i in range(50):
        start = time.time()
        
        envs.reset()
        for i in range(640 // n_envs):
            obs, r, d, info = envs.step([envs.action_space.sample() for i in range(n_envs)])
            
        run_times.append(time.time() - start)
    
    mean = np.mean(run_times)
    std = np.std(run_times)
    
    print(f'{mean*1000:.1f} ms ± {std*1000:.1f}')
    
    
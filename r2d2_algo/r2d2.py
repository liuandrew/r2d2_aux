
import torch
import torch.optim as optim
import gym
import gym_nav
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

from model import RNNQNetwork, linear_schedule
from storage import SequenceReplayBuffer
from r2d2_class import R2D2Agent
from args import get_args

import os
import sys
from pathlib import Path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from scheduler import archive_config_file

if __name__ == '__main__':
    args = get_args()
    env_id = args.env_id
    env_kwargs = args.env_kwargs
    learning_rate = args.learning_rate
    buffer_size = args.buffer_size
    total_timesteps = args.total_timesteps
    learning_starts = args.learning_starts
    train_frequency = args.train_frequency
    gamma = args.gamma
    tau = args.tau
    target_network_frequency = args.target_network_frequency

    start_e = args.start_e
    end_e = args.end_e
    exploration_fraction = args.exploration_fraction

    burn_in_length = args.burn_in_length
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    hidden_size = 64
    n_envs = args.n_envs
    
    seed = args.seed
    torch_deterministic = args.torch_deterministic
    checkpoint_interval = args.checkpoint_interval
    track = args.track
    exp_name = args.exp_name
    
    if exp_name == None:
        exp_name = env_id
    cuda = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
    
    
    if checkpoint_interval > 0:
        chk_folder = Path('saved_checkpoints/' + args.checkpoint_dir)/args.save_name
        chk_folder.mkdir(exist_ok=True, parents=True)
    
    run_name = f"{exp_name}__{seed}__{int(time.time())}"
    if track:
        import wandb
        
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f'runs/{run_name}')
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    
    agent = R2D2Agent(batch_size, burn_in_length, sequence_length,
                      gamma, tau, learning_rate, hidden_size,
                      device, buffer_size, learning_starts, train_frequency,
                      target_network_frequency, total_timesteps, start_e,
                      end_e, exploration_fraction, seed, n_envs,
                      False, env_id, env_kwargs, writer=writer, verbose=1)
    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    
    total_timesteps = total_timesteps // n_envs
    
    for t in range(total_timesteps):     
        #Training
        agent.collect(1)
        
        if agent.global_step > learning_starts:
            if t % train_frequency == 0:
                agent.update()
                
                if checkpoint_interval > 0 and (agent.global_update_step % checkpoint_interval == 0):
                    chk_path = chk_folder/f'{agent.global_update_step}.pt'
                    torch.save(agent.q_network, chk_path)
                
                #checkpoint
                #if args.checkpoint_interval > 0 and global_update_step % args.checkpoint_interval == 0:
                #   checkpoint_path = f'saved_checkpoints/{args.save_name}'
                #   ...
             
                        
    if args.save_name is not None:        
        #Save just the q_network which can be used to generate actions
        save_path = Path('saved_models/' + args.save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        save_path = save_path/f'{args.save_name}.pt'
        torch.save(agent.q_network, save_path)
        
    #For completeness, also save final checkpoint
    if checkpoint_interval > 0:
        chk_path = chk_folder/f'{agent.global_update_step}.pt'
        torch.save(agent.q_network, chk_path)
        
        #Code to save entire training history which can be reinitialized later
        # torch.save({
        #     'q_network': q_network,
        #     'target_network': target_network,
        #     'buffer': rb,
        #     'last_obs': obs,
        #     'last_rnn_hxs': rnn_hxs,
        #     'env': env,
        #     'global_step': global_step,
        #     'global_update_step': global_update_step
        # }, save_path)
        
    if args.config_file_name is not None:
        archive_config_file(args.config_file_name)
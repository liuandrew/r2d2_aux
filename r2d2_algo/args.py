import argparse
from ast import literal_eval
import os


#Andy: code used for passing environmental kwargs through command line
def isfloat(str):
    try:
        float(str)
        return True
    except:
        return False

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        print(values)
        for value in values:
            key, value = value.split('=')

            if value.isnumeric():
                value = int(value)
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif value == 'None':
                value = None
            elif isfloat(value):
                value = float(value)
            elif '[' in value:
                value = literal_eval(value)

            getattr(namespace, self.dest)[key] = value
            
class ParseList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, list())
        print(values)
        # for value in values:
        #     if '[' in value:
        #         value = literal_eval(value)
        #         getattr(namespace, self.dest).append(value)
        for value in values:
            if '[' in value:
                value = literal_eval(value)
            setattr(namespace, self.dest, value)
            
            
def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=None,
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    
    # File arguments 
    parser.add_argument("--save-name", type=str, default=None,
        help="name for model to be saved as")
    parser.add_argument("--save-dir", type=str, default='',
        help="subdirectory of runs/, saved_models/, saved_checkpoints/ to save to")
    parser.add_argument("--checkpoint-interval", type=int, default=0,
        help="interval of updates to save model checkpoints at. If 0, don't save checkpoints." + \
            "steps per checkpoint given by checkpoint_interval*train_frequency")
    parser.add_argument("--config-file-name", type=str, default=None,
        help="added from scheduler.py so that the main algo file knows to remove the config file on completion")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--n-envs", type=int, default=4,
        help='how many parallel vectorized envs to use (default: 4)')
    parser.add_argument("--dummy-vec-env", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if True use DummyVecEnv, if False use SubProcEnv")
    parser.add_argument("--total-timesteps", type=int, default=30000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size (note this is per env, so total will be n_envs*buffer_size)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--alpha", type=float, default=0.6,
        help="alpha hyperparameter in PER, determining how strongly priorities affect sampling. 0 for standard replay")
    parser.add_argument("--beta", type=float, default=0.4,
        help="beta hyperparameter in PER, determining importance sampling strength")
    parser.add_argument("--use-nstep-returns", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use all rewards in the returned sequences as td targets as opposed to 1 step fully bootstrapped targets")
    parser.add_argument("--use-segment-tree", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use segment trees for priority calculations")
    parser.add_argument("--target-network-frequency", type=int, default=64,
        help="the number of network updates it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the replay memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8,
        help="epsilon hyperparameter for Adam optimizer")
    
    #R2D2 args
    parser.add_argument("--burn-in-length", type=int, default=4,
        help="Length of timesteps prior to each batch sequence to burn-in so rnn_hidden_states are not stale")
    parser.add_argument("--sequence-length", type=int, default=8,
        help="Length of each batch sequence to use")
    # fmt: on
        
    parser.add_argument('--env-kwargs', nargs='*', action=ParseKwargs, default={})



    args = parser.parse_args()
    return args
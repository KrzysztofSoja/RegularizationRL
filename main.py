import os
import gym
import time
import argparse
import stable_baselines3 as sb
import sb3_contrib as sbc

from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.a2c import MlpPolicy as A2CMlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.ddpg import MlpPolicy as DDPGMlpPolicy
from stable_baselines3.td3 import MlpPolicy as TD3MlpPolicy
from sb3_contrib.tqc import MlpPolicy as TQCMlpPolicy

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks.cyclic_monitor import CyclicMonitor
from callbacks.neptune_callback import NeptuneCallback


ENVIRONMENT_NAMES = ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
POLICY = {'PPO': PPOMlpPolicy, 'A2C': A2CMlpPolicy, 'SAC': SACMlpPolicy, 'DDPG': DDPGMlpPolicy, 'TD3': TD3MlpPolicy,
          'TQC': TQCMlpPolicy}
MULTI_ENV = {'PPO', 'A2C'}

MAIN_DIR = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, 'logs'))
if not os.path.exists(MAIN_DIR):
    os.mkdir(MAIN_DIR)


def _make_env(env_name: str, rank: int, path_to_logs: str, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_name: the environment_name ID
    :param rank: index of the subprocess
    :param log_dir: Path to log directory.
    :param seed: the inital seed for RNG
    """

    def _init():
        env = gym.make(env_name)
        env.seed(seed + rank)
        env = CyclicMonitor(env, max_file_size=20, filename=os.path.join(path_to_logs, f'worker_{rank}'))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--make_video', default=False, action='store_true')
    args = parser.parse_args()

    assert args.env in ENVIRONMENT_NAMES, "Environments must be in environment list."
    assert args.algo in sb.__dict__ or args.algo in sbc.__dict__, "Algorithm name must be defined in stable_baselines3."

    if args.algo not in MULTI_ENV and args.workers != 1:
        print('Chosen algorithm don\'t support multi workers environment.')
        args.workers = 1

    print(f'Starting experiment with {args.algo} algorithm in {args.env} with {args.workers} for {args.steps} steps.')

    path_to_logs = os.path.join(MAIN_DIR, args.algo + '-' + args.env + '-' + str(time.time()).replace('.', ''))
    if not os.path.exists(path_to_logs):
        os.mkdir(path_to_logs)

    if args.algo in MULTI_ENV:
        train_env = DummyVecEnv([_make_env(args.env, i, path_to_logs) for i in range(args.workers)])
        eval_env = gym.make(args.env)
    else:
        train_env = CyclicMonitor(gym.make(args.env), max_file_size=20, filename=os.path.join(path_to_logs, f'all'))
        eval_env = gym.make(args.env)

    if args.algo == 'TQC':
        policy_kwargs = dict(n_critics=2, n_quantiles=25)
        model = sbc.TQC(TQCMlpPolicy, train_env, top_quantiles_to_drop_per_net=2, verbose=1,
                        policy_kwargs=policy_kwargs)
    else:
        model = sb.__dict__[args.algo](POLICY[args.algo], train_env, verbose=1)
    model.learn(total_timesteps=args.steps, callback=NeptuneCallback(model=model,
                                                                     experiment_name=args.env,
                                                                     neptune_account_name='nkrsi',
                                                                     project_name='rl-first-run',
                                                                     environment_name=args.env,
                                                                     log_dir=path_to_logs,
                                                                     make_video=args.make_video))
    model.save(os.path.join(path_to_logs, 'last_model.plk'))

import os
import gym
# mport pybulletgym
import time
import argparse
import torch
import random as rand

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks.cyclic_monitor import CyclicMonitor
from callbacks.neptune_callback import NeptuneCallback


import modificated_sb3 as sb
from modificated_sb3.ppo import MlpPolicy as PPOMlpPolicy
from modificated_sb3.a2c import MlpPolicy as A2CMlpPolicy
from modificated_sb3.sac import MlpPolicy as SACMlpPolicy
from modificated_sb3.ddpg import MlpPolicy as DDPGMlpPolicy
from modificated_sb3.td3 import MlpPolicy as TD3MlpPolicy
from modificated_sb3.tqc import MlpPolicy as TQCMlpPolicy

from common.torch_layers import create_mlp_with_dropout, MlpExtractorWithDropout, MlpExtractorWithManifoldMixup


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


def create_tags_list(args):
    tags = []
    for key, value in args.__dict__.items():
        if key in {'make_video_freq', 'validation_freq', 'validation_length', 'comment', 'use_neptune', 'steps',
                   'workers', 'algo', 'env'}:
            continue
        elif isinstance(value, bool) and value:
            tags.append(key)
        elif isinstance(value, str):
            tags.append(value)
        elif value:
            tags.append(f"{key}: {value}")
    return tags


def create_addition_params(args):
    parameters = {}
    parameters['steps'] = args.steps
    if args.algo in MULTI_ENV:
        parameters['workers'] = args.workers

    parameters['validation_freq'] = args.validation_freq
    parameters['validation_length'] = args.validation_length
    return parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Base experiment parameters
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    # Logger parameters
    parser.add_argument('--make_video_freq', type=int, default=0)
    parser.add_argument('--validation_freq', type=int, default=10_000)
    parser.add_argument('--validation_length', type=int, default=10)
    parser.add_argument('--use_neptune', default=False, action='store_true')
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--additional_tag', type=str, default=None)

    # Algorithm parameters
    parser.add_argument('--dropout_rate_critic', type=float, default=0)
    parser.add_argument('--dropout_rate_actor', type=float, default=0)
    parser.add_argument('--dropout_only_on_last_layer', type=bool, default=True)

    parser.add_argument('--weight_decay', type=float, default=False)
    parser.add_argument('--entropy_coefficient', type=float, default=False)
    parser.add_argument('--manifold_mixup_alpha', type=float, default=False)

    parser.add_argument('--gradient_penalty_actor', type=float, default=False)
    parser.add_argument('--gradient_penalty_actor_k', type=float, default=0)
    parser.add_argument('--gradient_penalty_critic', type=float, default=False)
    parser.add_argument('--gradient_penalty_critic_k', type=float, default=0)

    parser.add_argument('--use_sde', default=False, action='store_true')
    parser.add_argument('--device', default='cpu', type=str)

    args = parser.parse_args()

    # assert args.env in ENVIRONMENT_NAMES, "Environments must be in environment list."
    assert args.algo in sb.__dict__ or args.algo, "Algorithm name must be defined in stable_baselines3."
    assert 0 <= args.dropout_rate_critic < 1, "Dropout value must be from zero to one. "
    assert 0 <= args.dropout_rate_actor < 1, "Dropout value must be from zero to one. "
    assert args.manifold_mixup_alpha is False or args.manifold_mixup_alpha >= 0.0, \
        "Manifold mixup alpha must be positive real number."
    assert not (args.dropout_rate_critic > 0 and args.manifold_mixup_alpha), "Not implemented."
    assert not (args.dropout_rate_actor > 0 and args.manifold_mixup_alpha), "Not implemented."

    if args.seed is not None:
        torch.manual_seed(args.seed)
        rand.seed(args.seed)

    if args.algo not in MULTI_ENV and args.workers != 1:
        print('Chosen algorithm don\'t support multi workers environment.')
        args.workers = 1

    print(f'Starting experiment with {args.algo} algorithm in {args.env} with {args.workers} for {args.steps} steps.')
    if args.dropout_rate_critic:
        print(f'Algorithm using dropout with a value {args.dropout_rate_critic} on critic')

    if args.dropout_rate_actor:
        print(f'Algorithm using dropout with a value {args.dropout_rate_actor} on actor')

    if not args.dropout_only_on_last_layer:
        print("Dropout will be applicated on every layer in network.")

    if args.weight_decay:
        print(f'Algorithm using weight decay with a value {args.weight_decay}')

    if args.entropy_coefficient:
        print(f'Algorithm using entropy regularization with coefficient {args.entropy_coefficient}')

    if args.manifold_mixup_alpha:
        print(f"Algorithm use manifold mixup regularization with alpha parameter {args.manifold_mixup_alpha}")

    if args.gradient_penalty_actor:
        print(f"Algorithm use gradient penalty on actor with alpha parameter equal {args.gradient_penalty_actor}"
              f" and k equal {args.gradient_penalty_actor_k}")

    if args.gradient_penalty_critic:
        print(f"Algorithm use gradient penalty on actor with alpha parameter equal {args.gradient_penalty_critic}"
              f"and k equal {args.gradient_penalty_critic_k}")

    path_to_logs = os.path.join(MAIN_DIR, args.algo + '-' + args.env + '-' + str(time.time()).replace('.', ''))
    if not os.path.exists(path_to_logs):
        os.mkdir(path_to_logs)

    if args.algo in MULTI_ENV:
        train_env = DummyVecEnv([_make_env(args.env, i, path_to_logs, seed=args.seed) for i in range(args.workers)])
        eval_env = gym.make(args.env)
        eval_env.seed(args.seed)
    else:
        env = gym.make(args.env)
        env.seed(args.seed)
        train_env = CyclicMonitor(env, max_file_size=20, filename=os.path.join(path_to_logs, f'all'))
        eval_env = gym.make(args.env)
        eval_env.seed(args.seed)
        set_random_seed(args.seed)

    model_kwargs = dict()
    policy_kwargs = dict()

    if args.algo in {'A2C', 'PPO', 'SAC', 'TQC'}:
        model_kwargs['use_sde'] = args.use_sde

    if args.dropout_rate_critic or args.dropout_rate_actor:
        if args.algo in {'A2C', 'PPO'}:
            policy_kwargs['mlp_extractor_class'] = MlpExtractorWithDropout
            policy_kwargs['mpl_extractor_kwargs'] = {'dropout_rate_critic': args.dropout_rate_critic,
                                                     'dropout_rate_actor': args.dropout_rate_actor,
                                                     'dropout_only_on_last_layer': args.dropout_only_on_last_layer}
        else:
            policy_kwargs['create_network_function'] = create_mlp_with_dropout
            policy_kwargs['dropout_rate_critic'] = args.dropout_rate_critic
            policy_kwargs['dropout_rate_actor'] = args.dropout_rate_actor
            policy_kwargs['dropout_only_on_last_layer'] = args.dropout_only_on_last_layer

    if args.weight_decay:
        policy_kwargs['weight_decay'] = args.weight_decay

    if args.entropy_coefficient:
        if args.algo in {'A2C', 'PPO'}:
            model_kwargs['ent_coef'] = args.entropy_coefficient
        else:
            raise RuntimeError(f"{args.algo} hasn't entropy regularization")

    if args.manifold_mixup_alpha:
        if args.algo in {'A2C', 'PPO'}:
            policy_kwargs['mlp_extractor_class'] = MlpExtractorWithManifoldMixup
            policy_kwargs['mpl_extractor_kwargs'] = {'alpha': args.manifold_mixup_alpha, 'last_layer_mixup': 1}
        elif args.algo == 'TQC':
            raise RuntimeError(f"{args.algo} hasn't manifold mixup regularization")
        else:
            policy_kwargs['manifold_mixup_alpha'] = args.manifold_mixup_alpha

    if args.algo == 'TQC':
        policy_kwargs['n_critics'] = 2
        policy_kwargs['n_quantiles'] = 25

        print(model_kwargs)

        model = sb.TQC(TQCMlpPolicy, train_env, top_quantiles_to_drop_per_net=2, verbose=1,
                       policy_kwargs=policy_kwargs, device='cpu', **model_kwargs)
    else:
        model = sb.__dict__[args.algo](POLICY[args.algo], train_env, policy_kwargs=policy_kwargs, verbose=1,
                                       actor_gradient_penalty=args.gradient_penalty_actor,
                                       critic_gradient_penalty=args.gradient_penalty_critic,
                                       actor_gradient_penalty_k=args.gradient_penalty_actor_k,
                                       critic_gradient_penalty_k=args.gradient_penalty_critic_k,
                                       device=args.device, **model_kwargs)

    if args.use_neptune:
        callback_manager = NeptuneCallback(model=model, experiment_name=args.env, neptune_account_name='nkrsi',
                                           project_name='rl-first-run', environment_name=args.env, log_dir=path_to_logs,
                                           random_seed=args.seed, log_freq=100, evaluate_freq=args.validation_freq,
                                           make_video_freq=args.make_video_freq, evaluate_length=args.validation_length,
                                           model_parameter={**model_kwargs, **policy_kwargs, **create_addition_params(args)},
                                           comment=args.comment,
                                           tags=create_tags_list(args))
    else:
        callback_manager = None

    model.learn(total_timesteps=args.steps, callback=callback_manager)
    model.save(os.path.join(path_to_logs, 'last_model.plk'))

    print("Training done.")

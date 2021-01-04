import numpy as np
import stable_baselines3 as sb3

from stable_baselines3 import PPO, A2C, SAC, DQN, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise


class AbstractAlgoGetter:

    multi_processing = False

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class PPOGetter(AbstractAlgoGetter):

    multi_processing = True

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        model = PPO(sb3.ppo.MlpPolicy, environment, verbose)
        return model

    def __str__(self):
        return 'PPO'


class A2CGetter(AbstractAlgoGetter):

    multi_processing = True

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        model = A2C(sb3.a2c.MlpPolicy, environment, verbose)
        return model

    def __str__(self):
        return 'A2C'


class DQNGetter(AbstractAlgoGetter):

    multi_processing = False

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        model = DQN(sb3.dqn.MlpPolicy, environment, verbose)
        return model

    def __str__(self):
        return 'DQN'


class DDPGGetter(AbstractAlgoGetter):

    multi_processing = False

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        n_actions = environment.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG('MlpPolicy', environment, action_noise=action_noise, verbose=verbose)
        return model

    def __str__(self):
        return 'DDPG'


class SACGetter(AbstractAlgoGetter):

    multi_processing = False

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        model = SAC(sb3.sac.MlpPolicy, environment, verbose)
        return model

    def __str__(self):
        return 'SAC'


class TD3Gettter(AbstractAlgoGetter):

    multi_processing = False

    @staticmethod
    def get(environment, verbose=0, **kwargs):
        n_actions = environment.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3('MlpPolicy', environment, action_noise=action_noise, verbose=verbose)
        return model

    def __str__(self):
        return 'TD3'

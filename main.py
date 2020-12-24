import os
import gym
import csv
import random as rand
import numpy as np
import stable_baselines3 as sb3

from typing import Dict, Any, Tuple

from stable_baselines3 import PPO, A2C, SAC, DQN, DDPG, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
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


class MyCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=0):
        super(MyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def _load_logs(self) -> Tuple[float, float]:
        last_rewards = []
        for file_or_dir in os.listdir(self.log_dir):
            if os.path.isfile(file_or_dir) and file_or_dir[-len('.csv'):] == '.csv':
                with open(file_or_dir, 'r') as csv_file:
                    reader = csv.DictReader(csv_file)

                    for row in list(reader)[-2:]:
                        keys = list(row.keys())
                        try:
                            last_rewards.append(float(row[keys[0]]))
                        except ValueError:
                            pass
        if len(last_rewards) == 0:
            return None, None

        last_rewards = np.array(last_rewards)
        mean_reward, std_reward = np.mean(last_rewards), np.std(last_rewards)
        return mean_reward, std_reward

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward, std_reward = self._load_logs()

            if mean_reward is not None and mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
        return True


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, filename=env_id+'_'+str(rank))
        return env
    set_random_seed(seed)
    return _init


def run(env_name: str, algo_getter: AbstractAlgoGetter, steps: int, workers: int):
    if algo_getter.multi_processing:
        train_env = SubprocVecEnv([make_env(env_name, i) for i in range(workers)], start_method='fork')
        train_env.reset()
    else:
        train_env = make_env(env_name, 0)()
        train_env.reset()

    model = algo_getter.get(train_env, verbose=1)
    path_to_log = os.path.join(os.getcwd(), '{}'.format(str(algo_getter)))
    model.learn(total_timesteps=steps, callback=MyCallback(check_freq=1000, log_dir=path_to_log))
    return model


def show_model_action(env_name: str, model: BaseAlgorithm):
    env = gym.make(env_name)

    for i_episode in range(200):
        observation = env.reset()
        for t in range(100):
            env.render()
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


envs = ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
algo_getters = [PPOGetter(), A2CGetter(), DDPGGetter(), SACGetter(), TD3Gettter()]

for env in envs:
    for algo_getter in algo_getters:
        print("Experiment start in envinronment {} using {}". format(env, str(algo_getter)))
        run(env, algo_getter, 100, 8)


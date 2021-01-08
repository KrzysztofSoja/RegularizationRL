import os
import gym

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from getters import *
from callbacks.neptune_callback import NeptuneCallback
from callbacks.cyclic_monitor import CyclicMonitor


class Runner:

    def __init__(self, env_name: str, algo_getter: AbstractAlgoGetter, steps: int, workers: int):
        self.env_name = env_name
        self.algo_getter = algo_getter
        self.steps = steps
        self.workers = workers

    @staticmethod
    def _make_env(env_name: str, rank: int, log_dir: str, seed: int = 0):
        """
        Utility function for multiprocessed env.
        :param env_name: the environment ID
        :param rank: index of the subprocess
        :param log_dir: Path to log directory.
        :param seed: the inital seed for RNG
        """
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        def _init():
            env = gym.make(env_name)
            env.seed(seed + rank)
            env = CyclicMonitor(env, max_file_size=20, filename=os.path.join(log_dir, env_name+'_'+str(rank)))
            return env
        set_random_seed(seed)
        return _init

    def run(self):
        path_to_log = os.path.join(os.getcwd(), '{}'.format(str(self.algo_getter)))

        if algo_getter.multi_processing:
            train_env = SubprocVecEnv([Runner._make_env(self.env_name, i, path_to_log)
                                       for i in range(self.workers)], start_method='fork')
            train_env.reset()
        else:
            train_env = Runner._make_env(self.env_name, 0, path_to_log)()
            train_env.reset()

        eval_env = gym.make(self.env_name)
        model = algo_getter.get(train_env, verbose=1)
        model.learn(total_timesteps=self.steps, callback=NeptuneCallback(logs_freq=100,
                                                                         model=model,
                                                                         evaluate_freq=10_000,
                                                                         neptune_account_name="nkrsi",
                                                                         project_name="rl-first-run",
                                                                         experiment_name=model.__class__.__name__,
                                                                         eval_env=eval_env,
                                                                         log_dir=path_to_log))
        return model


envs = ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', 'Walker2d-v2', 'Hopper-v2', 'Humanoid-v2']
algo_getters = [PPOGetter(), A2CGetter(), DDPGGetter(), SACGetter(), TD3Gettter()]

for env in envs:
    for algo_getter in algo_getters:
        print("Experiment start in envinronment {} using {}". format(env, str(algo_getter)))
        runner = Runner(env, algo_getter, 100_000_000, 8)
        runner.run()
        break
    break

# ToDo:
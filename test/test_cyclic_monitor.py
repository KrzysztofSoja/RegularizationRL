import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


from callbacks.cyclic_monitor import CyclicMonitor

steps = 25000
env_name = 'CartPole-v1'


def _make_env(env_name: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    :param env_name: the environment ID
    :param rank: index of the subprocess
    :param log_dir: Path to log directory.
    :param seed: the inital seed for RNG
    """

    def _init():
        env = gym.make(env_name)
        env.seed(seed + rank)
        env = CyclicMonitor(env, max_file_size=20)
        return env

    set_random_seed(seed)
    return _init


train_env = DummyVecEnv([_make_env(env_name, i) for i in range(4)])

model = PPO(MlpPolicy, train_env, verbose=1)
model.learn(total_timesteps=steps)

validate_env = gym.make(env_name)
obs = validate_env.reset()
gain = 0

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = validate_env.step(action)
    gain += rewards
    if dones:
        break

print(gain)

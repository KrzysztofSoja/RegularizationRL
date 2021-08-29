import os
import sys
import gym
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
from modificated_sb3 import A2C, PPO, DDPG, SAC, TD3, TQC
from modificated_sb3.ppo import MlpPolicy as PPOMlpPolicy
from modificated_sb3.a2c import MlpPolicy as A2CMlpPolicy
from modificated_sb3.sac import MlpPolicy as SACMlpPolicy
from modificated_sb3.ddpg import MlpPolicy as DDPGMlpPolicy
from modificated_sb3.td3 import MlpPolicy as TD3MlpPolicy
from common.torch_layers import MlpExtractorWithDropout, create_mlp_with_dropout

# from stable_baselines3 import SAC
# from stable_baselines3.sac import MlpPolicy

class TestGradientPenalty(unittest.TestCase):

    def test_a2c(self):
        test_model = A2C(A2CMlpPolicy, 'CartPole-v1', verbose=1, actor_gradient_penalty=1.0,
                         critic_gradient_penalty=1.0,
                         actor_gradient_penalty_k=0.0,
                         critic_gradient_penalty_k=0.0)
        test_model.learn(10_000)

    def test_ppo(self):
        test_model = PPO(PPOMlpPolicy, 'CartPole-v1', verbose=1, actor_gradient_penalty=0.0,
                         critic_gradient_penalty=0.0,
                         actor_gradient_penalty_k=0.0,
                         critic_gradient_penalty_k=0.0)
        test_model.learn(10_000)

    def test_td3(self):
        test_model = TD3(TD3MlpPolicy, 'Pendulum-v0', verbose=1, actor_gradient_penalty=0.0,
                         critic_gradient_penalty=1.0,
                         actor_gradient_penalty_k=0.0,
                         critic_gradient_penalty_k=0.0)
        test_model.learn(10_000)

    def test_ddpg(self):
        test_model = DDPG(DDPGMlpPolicy, 'Pendulum-v0', verbose=1, actor_gradient_penalty=0.0,
                          critic_gradient_penalty=1.0,
                          actor_gradient_penalty_k=0.0,
                          critic_gradient_penalty_k=0.0)
        test_model.learn(10_000)

    def test_sac(self):
        import pybulletgym
        test_model = SAC(SACMlpPolicy, 'HalfCheetahPyBulletEnv-v0', verbose=1,  actor_gradient_penalty=1.0,
                         critic_gradient_penalty=1.0,
                         actor_gradient_penalty_k=0.0,
                         critic_gradient_penalty_k=0.0)
        test_model.learn(10_000)

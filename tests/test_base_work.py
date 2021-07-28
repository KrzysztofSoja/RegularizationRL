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
from modificated_sb3.tqc import MlpPolicy as TQCMlpPolicy


class TestWeightDecay(unittest.TestCase):
    """
    In this class, simple test is made. The assumption is to choose the parameter wd in such a way as to spoil
    the training. If a model with a large d parameter breaks, it means that setting it changes the behavior
    of the model.
    """

    TEST_ENV_BOX = 'CartPole-v1'
    TEST_ENV_CONTINUUM = 'Pendulum-v0'

    @staticmethod
    def __evaluate_model(model, env_name):
        env = gym.make(env_name)

        obs = env.reset()
        gain = 0
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            gain += reward
            if done: break

        return gain

    def test_a2c(self):
        print('Testing A2C model...')
        trained = A2C(A2CMlpPolicy, TestWeightDecay.TEST_ENV_BOX, verbose=0)
        random = A2C(A2CMlpPolicy, TestWeightDecay.TEST_ENV_BOX, verbose=0)
        trained.learn(5_000)

        gain_from_train = TestWeightDecay.__evaluate_model(trained, TestWeightDecay.TEST_ENV_BOX)
        gain_from_random = TestWeightDecay.__evaluate_model(random, TestWeightDecay.TEST_ENV_BOX)

        print(f"Gain from trained model - {gain_from_train}. Gain from random strategy - {gain_from_random}.")
        self.assertGreaterEqual(gain_from_train, gain_from_random)
        print('Test pass :)')

    def test_ppo(self):
        print('Testing  PPO model...')
        trained = PPO(PPOMlpPolicy, TestWeightDecay.TEST_ENV_BOX, verbose=0)
        random = PPO(PPOMlpPolicy, TestWeightDecay.TEST_ENV_BOX, verbose=0)
        trained.learn(5_000)

        gain_from_train = TestWeightDecay.__evaluate_model(trained, TestWeightDecay.TEST_ENV_BOX)
        gain_from_random = TestWeightDecay.__evaluate_model(random, TestWeightDecay.TEST_ENV_BOX)

        print(f"Gain from trained model - {gain_from_train}. Gain from random strategy - {gain_from_random}.")
        self.assertGreaterEqual(gain_from_train, gain_from_random)
        print('Test pass :)')

    # ToDo:
    def test_ddpg(self):
        """
        This model is too strong to make its weaker by bad value of weight decay.
        """

        print('Testing weight decay in DDPG model...')
        model = DDPG(DDPGMlpPolicy, TestWeightDecay.TEST_ENV_CONTINUUM,
                     policy_kwargs={'weight_decay': 1.0}, verbose=0)   # This is too strong! It is creepy.
        self.assertIn("weight_decay: 1.0", str(model.policy.actor.optimizer))
        self.assertIn("weight_decay: 1.0", str(model.policy.critic.optimizer))
        print('Test pass :)')

    def test_sac(self):
        print('Testing SAC model...')
        trained = SAC(SACMlpPolicy, TestWeightDecay.TEST_ENV_CONTINUUM, verbose=0)  # This is too strong! It is creepy.
        random = SAC(SACMlpPolicy, TestWeightDecay.TEST_ENV_CONTINUUM, verbose=0)
        trained.learn(10_000)

        gain_from_train = TestWeightDecay.__evaluate_model(trained, TestWeightDecay.TEST_ENV_CONTINUUM)
        gain_from_random = TestWeightDecay.__evaluate_model(random, TestWeightDecay.TEST_ENV_CONTINUUM)

        print(f"Gain from trained model - {gain_from_train}. Gain from random strategy - {gain_from_random}.")
        self.assertGreaterEqual(gain_from_train, gain_from_random)
        print('Test pass :)')

    def test_td3(self):
        print('Testing TD3 model...')
        trained = TD3(TD3MlpPolicy, TestWeightDecay.TEST_ENV_CONTINUUM, verbose=0)  # This is too strong! It is creepy.
        random = TD3(TD3MlpPolicy, TestWeightDecay.TEST_ENV_CONTINUUM, verbose=0)
        trained.learn(10_000)

        gain_from_train = TestWeightDecay.__evaluate_model(trained, TestWeightDecay.TEST_ENV_CONTINUUM)
        gain_from_random = TestWeightDecay.__evaluate_model(random, TestWeightDecay.TEST_ENV_CONTINUUM)

        print(f"Gain from trained model - {gain_from_train}. Gain from random strategy - {gain_from_random}.")
        self.assertGreaterEqual(gain_from_train, gain_from_random)
        print('Test pass :)')

    # ToDo
    def test_tqc(self):
        """
        This model is too strong to make its weaker by bad value of weight decay.
        """

        print('Testing weight decay in TQC model...')
        model = TQC(TQCMlpPolicy, TestWeightDecay.TEST_ENV_CONTINUUM,
                    policy_kwargs={'weight_decay': 1.0}, verbose=0)  # This is too strong! It is creepy.
        self.assertIn("weight_decay: 1.0", str(model.policy.actor.optimizer))
        self.assertIn("weight_decay: 1.0", str(model.policy.critic.optimizer))
        print('Test pass :)')

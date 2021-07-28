import os
import sys
import gym
import unittest

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir)))
from common.torch_layers import MlpExtractorWithManifoldMixup
from modificated_sb3 import A2C, PPO, DDPG, SAC, TD3, TQC
from modificated_sb3.ppo import MlpPolicy as PPOMlpPolicy
from modificated_sb3.a2c import MlpPolicy as A2CMlpPolicy
from modificated_sb3.sac import MlpPolicy as SACMlpPolicy
from modificated_sb3.ddpg import MlpPolicy as DDPGMlpPolicy
from modificated_sb3.td3 import MlpPolicy as TD3MlpPolicy
from modificated_sb3.tqc import MlpPolicy as TQCMlpPolicy


class TestManifoldMixup(unittest.TestCase):

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
        print('Testing manifold mixup in A2C model...')
        test_model = A2C(A2CMlpPolicy, TestManifoldMixup.TEST_ENV_BOX,
                         policy_kwargs={'mlp_extractor_class': MlpExtractorWithManifoldMixup,
                                        'mpl_extractor_kwargs': {'alpha': 0.1, 'last_layer_mixup': 1}}, verbose=0)
        random = A2C(A2CMlpPolicy, TestManifoldMixup.TEST_ENV_BOX, verbose=0)

        test_model.learn(10_000)
        gain_from_test = TestManifoldMixup.__evaluate_model(test_model, TestManifoldMixup.TEST_ENV_BOX)
        gain_from_random = TestManifoldMixup.__evaluate_model(random, TestManifoldMixup.TEST_ENV_BOX)

        print(f"Gain from test model - {gain_from_test}.")
        self.assertGreaterEqual(gain_from_test, gain_from_random)
        print('Test pass :)')

    def test_ppo(self):
        print('Testing manifold mixup in PPO model...')
        test_model = PPO(PPOMlpPolicy, TestManifoldMixup.TEST_ENV_BOX,
                         policy_kwargs={'mlp_extractor_class': MlpExtractorWithManifoldMixup,
                                        'mpl_extractor_kwargs': {'alpha': 0.1, 'last_layer_mixup': 1}}, verbose=0)
        random = PPO(PPOMlpPolicy, TestManifoldMixup.TEST_ENV_BOX, verbose=0)

        test_model.learn(10_000)
        gain_from_test = TestManifoldMixup.__evaluate_model(test_model, TestManifoldMixup.TEST_ENV_BOX)
        gain_from_random = TestManifoldMixup.__evaluate_model(random, TestManifoldMixup.TEST_ENV_BOX)

        print(f"Gain from test model - {gain_from_test}.")
        self.assertGreaterEqual(gain_from_test, gain_from_random)
        print('Test pass :)')

    def test_td3(self):
        print('Testing manifold mixup in TD3 model...')
        test_model = TD3(TD3MlpPolicy, TestManifoldMixup.TEST_ENV_CONTINUUM,
                         policy_kwargs={'manifold_mixup_alpha': 0.1}, verbose=0)
        print(test_model.policy)
        random = TD3(TD3MlpPolicy, TestManifoldMixup.TEST_ENV_CONTINUUM, verbose=0)

        test_model.learn(10_000)
        gain_from_test = TestManifoldMixup.__evaluate_model(test_model, TestManifoldMixup.TEST_ENV_CONTINUUM)
        gain_from_random = TestManifoldMixup.__evaluate_model(random, TestManifoldMixup.TEST_ENV_CONTINUUM)

        print(f"Gain from test model - {gain_from_test}.")
        self.assertGreaterEqual(gain_from_test, gain_from_random)
        print('Test pass :)')

    def test_ddpg(self):
        print('Testing manifold mixup in DDPG model...')
        test_model = DDPG(DDPGMlpPolicy, TestManifoldMixup.TEST_ENV_CONTINUUM,
                          policy_kwargs={'manifold_mixup_alpha': 0.1}, verbose=0)
        random = DDPG(DDPGMlpPolicy, TestManifoldMixup.TEST_ENV_CONTINUUM, verbose=0)

        test_model.learn(10_000)
        gain_from_test = TestManifoldMixup.__evaluate_model(test_model, TestManifoldMixup.TEST_ENV_CONTINUUM)
        gain_from_random = TestManifoldMixup.__evaluate_model(random, TestManifoldMixup.TEST_ENV_CONTINUUM)

        print(f"Gain from test model - {gain_from_test}.")
        self.assertGreaterEqual(gain_from_test, gain_from_random)
        print('Test pass :)')

    def test_sac(self):
        print('Testing manifold mixup in SAC model...')
        test_model = SAC(SACMlpPolicy, TestManifoldMixup.TEST_ENV_CONTINUUM,
                          policy_kwargs={'manifold_mixup_alpha': 0.1}, verbose=0)
        random = SAC(SACMlpPolicy, TestManifoldMixup.TEST_ENV_CONTINUUM, verbose=0)

        test_model.learn(10_000)
        gain_from_test = TestManifoldMixup.__evaluate_model(test_model, TestManifoldMixup.TEST_ENV_CONTINUUM)
        gain_from_random = TestManifoldMixup.__evaluate_model(random, TestManifoldMixup.TEST_ENV_CONTINUUM)

        print(f"Gain from test model - {gain_from_test}. Gain from random strategy - {gain_from_random}")
        self.assertGreaterEqual(gain_from_test, gain_from_random)
        print('Test pass :)')






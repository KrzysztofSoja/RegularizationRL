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
from common.torch_layers import MlpExtractorWithDropout, create_mlp_with_dropout


class BaseTestDropout(unittest.TestCase):
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
        print('Testing dropout in A2C model...')
        test_model = A2C(A2CMlpPolicy, BaseTestDropout.TEST_ENV_BOX,
                         policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                                        'mpl_extractor_kwargs': {'dropout_rate': 0.5}}, verbose=0)
        test_model.learn(10_000)
        target_model = A2C(A2CMlpPolicy, BaseTestDropout.TEST_ENV_BOX, verbose=0)
        target_model.learn(10_000)

        gain_from_test = BaseTestDropout.__evaluate_model(test_model, BaseTestDropout.TEST_ENV_BOX)
        gain_from_target = BaseTestDropout.__evaluate_model(target_model, BaseTestDropout.TEST_ENV_BOX)

        print(f"Gain from test model - {gain_from_test}. Gain from target model - {gain_from_target}.")
        self.assertGreaterEqual(gain_from_target/gain_from_test, 2)
        self.assertIn("Dropout(p=0.5,", str(test_model.policy))
        print('Test pass :)')

    def test_ppo(self):
        print('Testing weight decay in PPO model...')
        test_model = PPO(PPOMlpPolicy, "CartPole-v1",
                         policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                                        'mpl_extractor_kwargs': {'dropout_rate': 0.8}}, verbose=0)

        test_model.learn(10_000)
        target_model = PPO(PPOMlpPolicy, BaseTestDropout.TEST_ENV_BOX, verbose=0)
        target_model.learn(10_000)

        gain_from_test = BaseTestDropout.__evaluate_model(test_model, BaseTestDropout.TEST_ENV_BOX)
        gain_from_target = BaseTestDropout.__evaluate_model(target_model, BaseTestDropout.TEST_ENV_BOX)

        print(f"Gain from test model - {gain_from_test}. Gain from target model - {gain_from_target}.")
        self.assertIn("Dropout(p=0.8,", str(test_model.policy))
        self.assertGreaterEqual(gain_from_target / gain_from_test, 2)
        print('Test pass :)')

    def test_ddpg(self):
        """
        This model is too strong to make its weaker by bad value of dropout.
        """

        print('Testing weight decay in DDPG model...')
        model = DDPG(DDPGMlpPolicy, BaseTestDropout.TEST_ENV_CONTINUUM,
                     policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                    'dropout_rate': 0.5}, verbose=0)   # This is too strong! It is creepy.

        self.assertIn("Dropout(p=0.5,", str(model.policy))
        print('Test pass :)')

    def test_sac(self):
        """
        This model is too strong to make its weaker by bad value of weight decay.
        """

        print('Testing weight decay in DDPG model...')
        model = SAC(SACMlpPolicy, BaseTestDropout.TEST_ENV_CONTINUUM,
                    policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                   'dropout_rate': 0.5}, verbose=0)   # This is too strong! It is creepy.
        self.assertIn("Dropout(p=0.5,", str(model.policy))
        print('Test pass :)')

    def test_td3(self):
        """
        This model is too strong to make its weaker by bad value of weight decay.
        """

        print('Testing weight decay in TD3 model...')
        model = TD3(TD3MlpPolicy, BaseTestDropout.TEST_ENV_CONTINUUM,
                    policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                   'dropout_rate': 0.5}, verbose=0)  # This is too strong! It is creepy.

        self.assertIn("Dropout(p=0.5,", str(model.policy))
        print('Test pass :)')

    def test_tqc(self):
        """
        This model is too strong to make its weaker by bad value of weight decay.
        """

        print('Testing weight decay in TQC model...')
        model = TQC(TQCMlpPolicy, BaseTestDropout.TEST_ENV_CONTINUUM,
                    policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                   'dropout_rate': 0.5}, verbose=0)  # This is too strong! It is creepy.
        self.assertIn("Dropout(p=0.5,", str(model.policy))
        print('Test pass :)')


class TestDropout(unittest.TestCase):

    TEST_ENV_BOX = 'CartPole-v1'
    TEST_ENV_CONTINUUM = 'Pendulum-v0'

    def test_a2c(self):
        model = A2C(A2CMlpPolicy, TestDropout.TEST_ENV_BOX,
                    policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                                   'mpl_extractor_kwargs': {'dropout_rate_actor': 0.1,
                                                            'dropout_rate_critic': 0,
                                                            'dropout_only_on_last_layer': False}}, verbose=1)

        print(model.policy)
        model.learn(10_000)

    def test_ppo(self):
        model = PPO(PPOMlpPolicy, TestDropout.TEST_ENV_BOX,
                    policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                                   'mpl_extractor_kwargs': {'dropout_rate_actor': 0.2,
                                                            'dropout_rate_critic': 0.2,
                                                            'dropout_only_on_last_layer': True}}, verbose=1)

        print(model.policy)
        model.learn(10_000)

    def test_td3(self):
        model = TD3(TD3MlpPolicy, TestDropout.TEST_ENV_CONTINUUM,
                    policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                   'dropout_rate_actor': 0.5,
                                   'dropout_rate_critic': 0.5}, verbose=1)
        print(model.policy)
        model.learn(10_000)


    def test_sac(self):
        model = TD3(TD3MlpPolicy, TestDropout.TEST_ENV_CONTINUUM,
                    policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                   'dropout_rate_actor': 0.1,
                                   'dropout_rate_critic': 0,
                                   'dropout_only_on_last_layer': True}, verbose=1)
        print(model.policy)
        model.learn(10_000)

from stable_baselines3 import PPO

try:
    from .a2c import ActorCriticPolicy
except ImportError:
    from a2c import ActorCriticPolicy
from common.torch_layers import MlpExtractorWithDropout


MlpPolicy = ActorCriticPolicy


if __name__ == '__main__':
    model = PPO(MlpPolicy, "CartPole-v1",
                policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                               'mpl_extractor_kwargs': {'dropout_rate': 0.5}},
                verbose=1)
    model.learn(10_000)

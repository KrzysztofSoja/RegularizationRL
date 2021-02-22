from modificated_sb3.td3 import TD3Policy
from stable_baselines3.ddpg import DDPG
from common.torch_layers import create_mlp_with_dropout


MlpPolicy = TD3Policy
DDPGPolicy = TD3Policy

if __name__ == '__main__':
    model = DDPG(MlpPolicy, "Pendulum-v0",
                 policy_kwargs={'create_network_function': create_mlp_with_dropout,
                                'dropout_rate': 0.5},
                 verbose=1)
    model.learn(10_000)

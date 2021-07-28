import gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3 import TD3 as BaseTD3
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
from stable_baselines3.td3.policies import TD3Policy as BaseTD3Policy
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy

from typing import Any, Callable, Dict, List, Optional, Type, Union

from common.policies import ContinuousCriticWithManifoldMixup, ContinuousCriticWithDropout
from common.torch_layers import create_mlp_with_dropout


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        create_network: callable = create_mlp,
        dropout_rate: Optional[float] = .5,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        if create_network.__name__ == create_mlp_with_dropout.__name__:
            actor_net = create_network(features_dim, action_dim, net_arch, activation_fn, dropout_rate=dropout_rate,
                                       squash_output=True)
        else:
            actor_net = create_network(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        self.mu = nn.Sequential(*actor_net)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.mu(features)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic=deterministic)


class TD3Policy(BaseTD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        dropout_rate: Optional[float] = None,
        weight_decay: Optional[float] = 0.0,
        manifold_mixup_alpha: Optional[float] = 0.0,
        last_layer_mixup: Optional[int] = 1,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        create_network_function: callable = create_mlp,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        assert not (dropout_rate is not None and dropout_rate > 0 and manifold_mixup_alpha > 0), 'Not implemented.'

        if weight_decay != 0.0:
            if optimizer_kwargs is None:
                optimizer_kwargs = dict()
            optimizer_kwargs['weight_decay'] = weight_decay

        self.weight_decay = weight_decay
        self.create_network_function = create_network_function
        if self.create_network_function.__name__ == create_mlp_with_dropout.__name__:
            self.dropout_rate = .5 if dropout_rate is None else dropout_rate
        self.manifold_mixup_alpha = manifold_mixup_alpha
        self.last_layer_mixup = last_layer_mixup

        super(TD3Policy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        if self.create_network_function.__name__ == create_mlp_with_dropout.__name__:
            actor_kwargs['dropout_rate'] = self.dropout_rate

        return Actor(create_network=self.create_network_function, **actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Any:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self.create_network_function.__name__ == create_mlp_with_dropout.__name__:
            critic_kwargs['dropout_rate'] = self.dropout_rate
            return ContinuousCriticWithDropout(create_network=self.create_network_function, **critic_kwargs).to(self.device)
        elif self.manifold_mixup_alpha > 0:
            critic_kwargs['alpha'] = self.manifold_mixup_alpha
            critic_kwargs['last_layer_mixup'] = self.last_layer_mixup
            return ContinuousCriticWithManifoldMixup(**critic_kwargs).to(
                self.device)
        else:
            return super(TD3Policy, self).make_critic(features_extractor)


class TD3(BaseTD3):

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values   # ToDo: Target_Q_values

            if isinstance(self.critic, ContinuousCriticWithManifoldMixup):
                # Get current Q-values estimates for each critic network
                current_q_values_and_new_targets = self.critic(replay_data.observations, replay_data.actions,
                                                               target_q_values)
                # Compute critic loss with new targets
                critic_loss = sum([F.mse_loss(current_q, new_target) for current_q, new_target in
                                   current_q_values_and_new_targets])
            else:
                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                 # Compute critic loss
                critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            temp_actor_losses = np.mean(actor_losses)
            logger.record("train/actor_loss", temp_actor_losses)
            if np.isinf(temp_actor_losses) or np.isnan(temp_actor_losses):
                raise Exception('Gradient KABUM! actor_loss.')
        temp_critic_losses = np.mean(critic_losses)
        logger.record("train/critic_loss", temp_critic_losses)
        if np.isinf(temp_critic_losses) or np.isnan(temp_critic_losses):
            raise Exception('Gradient KABUM! critic_losses.')


MlpPolicy = TD3Policy


if __name__ == '__main__':
    model = TD3(TD3Policy, "Pendulum-v0",
                policy_kwargs={'create_network_function': create_mlp_with_dropout,
                               'dropout_rate': 0.1,
                               'weight_decay': 0.5},
                verbose=1)

    print(model.policy.actor.optimizer)
    model.learn(10_000)

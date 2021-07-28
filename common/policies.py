import gym
import random as rand
import torch as th
import numpy as np
import torch.nn as nn

from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from typing import List, Type, Tuple, Optional, Union

from .torch_layers import create_mlp_with_dropout


def mixup_data(x, y, alpha, device: Union[th.device, str] = "auto"):
    ''' Compute the mixup data. Return mixed inputs, pairs of targets, and lambda '''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = th.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ContinuousCriticWithDropout(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        dropout_rate: Optional[float] = .5,
        create_network: callable = create_mlp,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):

        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            if create_network.__name__ == create_mlp_with_dropout.__name__:
                q_net = create_network(features_dim + action_dim, 1, net_arch, activation_fn, dropout_rate=dropout_rate)
            else:
                q_net = create_network(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class SingleContinuousCritic(nn.Module):
    """
    Part of ContinuousCritic.
    """

    def __init__(self,
                 net_arch: List[int],
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 last_layer_mixup: int = 1,
                 alpha: float = 0.1,
                 device: Union[th.device, str] = 'cpu'):
        super(SingleContinuousCritic, self).__init__()
        self.last_layer_mixup = last_layer_mixup
        self.alpha = alpha
        self.device = device

        assert len(net_arch) == 2, "Depth of this critic isn't regularized. List must have length equal two."

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=net_arch[0], bias=True),
            activation_fn(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=net_arch[0], out_features=net_arch[1], bias=True),
            activation_fn(),
        )
        self.layer3 = nn.Linear(in_features=net_arch[1], out_features=1, bias=True)

    def forward(self, x: th.Tensor, y: th.Tensor = None) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        if y is None:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
        else:
            layer_mix = rand.randint(0, self.last_layer_mixup)
            if layer_mix == 0:
                x, y_a, y_b, lam = mixup_data(x, y, self.alpha, self.device)
            x = self.layer1(x)
            if layer_mix == 1:
                x, y_a, y_b, lam = mixup_data(x, y, self.alpha, self.device)
            x = self.layer2(x)
            # Mixup height layer can make problems.
            if layer_mix == 2:
                x, y_a, y_b, lam = mixup_data(x, y, self.alpha, self.device)
            x = self.layer3(x)

            lam = th.tensor(lam).to(self.device)
            y = y_a * lam.expand_as(y_a) + y_b * (1 - lam.expand_as(y_b))
            return x, y


class ContinuousCriticWithManifoldMixup(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        last_layer_mixup: int = 1,
        alpha: float = 0.1,
        device: Union[th.device, str] = 'cpu'
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = SingleContinuousCritic(net_arch, activation_fn, last_layer_mixup, alpha, device)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor, ground_truth_q: Optional[th.Tensor] = None) \
            -> Tuple[Union[th.Tensor, Tuple[th.Tensor, th.Tensor]], ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)

        if ground_truth_q is None:
            return tuple(q_net(qvalue_input) for q_net in self.q_networks)
        else:
            return tuple(q_net(qvalue_input, ground_truth_q) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor, ground_truth_q: Optional[th.Tensor] = None) \
            -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)

        qvalue_input = th.cat([features, actions], dim=1)
        if ground_truth_q is None:
            return self.q_networks[0](qvalue_input)
        else:
            return self.q_networks[0](qvalue_input, ground_truth_q)

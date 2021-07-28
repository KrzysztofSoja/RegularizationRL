import gym
import random as rand
import numpy as np
import torch as th
import torch.nn as nn

from itertools import zip_longest
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BaseModel

from typing import List, Type, Union, Dict, Tuple, Optional


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


class MlpExtractorWithDropout(nn.Module):
    """
    Adding dropout to MlpExtractor from stable-baselines3.

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    :param dropout_rate: Probability of an perceptron to be zeroed.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        dropout_rate: float = 0.5
    ):
        super(MlpExtractorWithDropout, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        last_layer_dim_shared = feature_dim

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                shared_net.append(nn.Dropout(p=dropout_rate))                          # ToDo: Dodać parametryzacje
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                policy_net.append(nn.Dropout(p=dropout_rate))
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                value_net.append(nn.Dropout(p=dropout_rate))
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


def create_mlp_with_dropout(
    input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False, dropout_rate: float = 0.5) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function and dropout.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param dropout_rate: probability of an element to be zeroed in Dropout layer. Default: 0.5
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn(), nn.Dropout(p=dropout_rate)]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
        modules.append(nn.Dropout(p=dropout_rate))

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# ToDo: Zmienić nazwę tak, żeby było widać, do jakich agentów należy ten extraktor.
class MlpExtractorWithManifoldMixup(nn.Module):
    """
    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        last_layer_mixup: int = 1,
        alpha: float = 0.1,
        device: Union[th.device, str] = "auto",
    ):
        super(MlpExtractorWithManifoldMixup, self).__init__()

        self.last_layer_mixup = max(last_layer_mixup, 2)
        if last_layer_mixup > 2:
            print(f"WARNING!!! Maximal possible layer of mixup is 2. Instead of, given value is {last_layer_mixup}. "
                  f"Set 2 as this variable.")
        self.alpha = alpha
        self.device = device

        # Save dim, used to create the distributions
        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential().to(device)
        self.policy_net = nn.Sequential(nn.Linear(in_features=4, out_features=64, bias=True),
                                        nn.Tanh(),
                                        nn.Linear(in_features=64, out_features=64, bias=True),
                                        nn.Tanh()).to(device)
        self.value_net_layer_1 = nn.Sequential(nn.Linear(in_features=4, out_features=64, bias=True),
                                               nn.Tanh()).to(device)
        self.value_net_layer_2 = nn.Sequential(nn.Linear(in_features=64, out_features=64, bias=True),
                                               nn.Tanh()).to(device)

    def __mixup_forward_values(self, features: th.Tensor, ground_truth_values: th.Tensor) \
            -> Tuple[th.Tensor, th.Tensor]:
        layer_mix = rand.randint(0, self.last_layer_mixup)   # ToDo: Dodać parametr.

        if layer_mix == 0:
            features, values_a, values_b, lam = mixup_data(features, ground_truth_values, self.alpha, self.device)
        features = self.value_net_layer_1(features)
        if layer_mix == 1:
            features, values_a, values_b, lam = mixup_data(features, ground_truth_values, self.alpha, self.device)
        features = self.value_net_layer_2(features)
        # Mixup height layer can make problems.
        if layer_mix == 2:
            features, values_a, values_b, lam = mixup_data(features, ground_truth_values, self.alpha, self.device)

        lam = th.tensor(lam).to(self.device)
        ground_truth_values = values_a * lam.expand_as(values_a) + values_b * (1 - lam.expand_as(values_b))
        return features, ground_truth_values

    def forward(self, features: th.Tensor, ground_truth_values: Optional[th.Tensor] = None) -> \
            Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor, th.Tensor]]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        policy = self.policy_net(shared_latent)

        if ground_truth_values is None:
            value = self.value_net_layer_1(shared_latent)
            value = self.value_net_layer_2(value)
            return policy, value
        else:
            value, ground_truth_values = self.__mixup_forward_values(shared_latent, ground_truth_values)
            return policy, value, ground_truth_values

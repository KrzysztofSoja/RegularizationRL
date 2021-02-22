import gym
import torch as th
import torch.nn as nn

from sb3_contrib.tqc import TQC
from sb3_contrib.tqc.policies import MlpPolicy as BaseTQCPolicy
from sb3_contrib.tqc.policies import Actor as BaseActor
from typing import Any, Callable, Dict, List, Optional, Type
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy, create_sde_features_extractor
from stable_baselines3.common.preprocessing import get_action_dim

from common.torch_layers import create_mlp_with_dropout

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)


class Actor(BaseActor):
    """
    Actor network (policy) for TQC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        create_network: callable = create_mlp,
        dropout_rate: Optional[float] = .5,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(BaseActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        if create_network.__name__ == create_mlp_with_dropout.__name__:
            latent_pi_net = create_network(features_dim, -1, net_arch, activation_fn, dropout_rate=dropout_rate)
        else:
            latent_pi_net = create_network(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate feature extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                    features_dim, sde_net_arch, activation_fn
                )

            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)




class Critic(BaseModel):
    """
    Critic network (q-value function) for TQC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
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
        create_network: callable = create_mlp,
        dropout_rate: Optional[float] = .5,
        normalize_images: bool = True,
        n_quantiles: int = 25,
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
        self.q_networks = []
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.quantiles_total = n_quantiles * n_critics

        for i in range(n_critics):
            if create_network.__name__ == create_mlp_with_dropout.__name__:
                qf_net = create_network(features_dim + action_dim, n_quantiles, net_arch, activation_fn,
                                    dropout_rate=dropout_rate)
            else:
                qf_net = create_network(features_dim + action_dim, n_quantiles, net_arch, activation_fn)
            qf_net = nn.Sequential(*qf_net)
            self.add_module(f"qf{i}", qf_net)
            self.q_networks.append(qf_net)

    def forward(self, obs: th.Tensor, action: th.Tensor) -> List[th.Tensor]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, action], dim=1)
        quantiles = th.stack(tuple(qf(qvalue_input) for qf in self.q_networks), dim=1)
        return quantiles


class TQCPolicy(BaseTQCPolicy):
    """
        Policy class (with both actor and critic) for TQC.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule (could be constant)
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: Activation function
        :param use_sde: Whether to use State Dependent Exploration or not
        :param log_std_init: Initial value for the log standard deviation
        :param sde_net_arch: Network architecture for extracting features
            when using gSDE. If None, the latent features from the policy will be used.
            Pass an empty list to use the states as features.
        :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
        :param features_extractor_class: Features extractor to use.
        :param features_extractor_kwargs: Keyword arguments
            to pass to the feature extractor.
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param optimizer_class: The optimizer to use,
            ``th.optim.Adam`` by default
        :param optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
        :param share_features_extractor: Whether to share or not the features extractor
            between the actor and the critic (this saves computation time)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            create_network_function: callable = create_mlp,
            dropout_rate: Optional[float] = None,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_quantiles: int = 25,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):

        self.create_network_function = create_network_function
        if self.create_network_function.__name__ == create_mlp_with_dropout.__name__:
            self.dropout_rate = .5 if dropout_rate is None else dropout_rate

        super(BaseTQCPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [256, 256]
            else:
                net_arch = []

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "sde_net_arch": sde_net_arch,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        tqc_kwargs = {
            "n_quantiles": n_quantiles,
            "n_critics": n_critics,
            "net_arch": critic_arch,
            "share_features_extractor": share_features_extractor,
        }
        self.critic_kwargs.update(tqc_kwargs)
        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        if self.create_network_function.__name__ == create_mlp_with_dropout.__name__:
            actor_kwargs['dropout_rate'] = self.dropout_rate
        return Actor(create_network=self.create_network_function, **actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Critic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self.create_network_function.__name__ == create_mlp_with_dropout.__name__:
            critic_kwargs['dropout_rate'] = self.dropout_rate
        return Critic(create_network=self.create_network_function, **critic_kwargs).to(self.device)


MlpPolicy = TQCPolicy

if __name__ == '__main__':
    model = TQC(TQCPolicy, "Pendulum-v0",
                policy_kwargs={'create_network_function': create_mlp_with_dropout,
                               'dropout_rate': 0.5},
                verbose=1)
    model.learn(10_000)


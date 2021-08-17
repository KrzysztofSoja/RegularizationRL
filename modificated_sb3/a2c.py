import torch as th
import numpy as np

from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3 import A2C as BaseA2C
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.policies import ActorCriticPolicy as BaseActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.utils import explained_variance

from typing import Dict, Optional, Type, Union, Any, Callable, Tuple

from common.torch_layers import MlpExtractorWithDropout, MlpExtractorWithManifoldMixup
from common.gradient_penalty import gradient_penalty_actor_critic

__all__ = ['A2C', 'ActorCriticPolicy', 'MlpPolicy']


class ActorCriticPolicy(BaseActorCriticPolicy):

    def __init__(self, *args, mlp_extractor_class: Union[Type[MlpExtractor], Type[MlpExtractorWithDropout]] = None,
                 mpl_extractor_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        self.mlp_extractor_class = MlpExtractor if mlp_extractor_class is None else mlp_extractor_class
        self.mpl_extractor_kwargs = dict() if mpl_extractor_kwargs is None else mpl_extractor_kwargs

        super(ActorCriticPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).

        self.mlp_extractor = self.mlp_extractor_class(
            self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device,
            **self.mpl_extractor_kwargs
        )

    def _get_latent(self, obs: th.Tensor, ground_truth_values: Optional[th.Tensor] = None) \
            -> Union[Tuple[th.Tensor, th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)

        if ground_truth_values is None:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            latent_pi, latent_vf, ground_truth_values = self.mlp_extractor(features, ground_truth_values)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)

        if ground_truth_values is None:
            return latent_pi, latent_vf, latent_sde
        else:
            return latent_pi, latent_vf, latent_sde, ground_truth_values

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, returns: Optional[th.Tensor] = None) \
            -> Union[Tuple[th.Tensor, th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :param returns: set, if manifold mixup is using
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        if returns is None:
            latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        else:
            latent_pi, latent_vf, latent_sde, ground_truth_values = self._get_latent(obs, returns)

        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        if returns is None:
            return values, log_prob, distribution.entropy()
        else:
            return values, log_prob, distribution.entropy(), ground_truth_values


class A2C(BaseA2C):

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable] = 7e-4,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rms_prop_eps: float = 1e-5,
            use_rms_prop: bool = True,
            use_sde: bool = False,
            actor_gradient_penalty: float = 0.0,
            critic_gradient_penalty: float = 0.0,
            actor_gradient_penalty_k: float = 1.0,
            critic_gradient_penalty_k: float = 1.0,
            sde_sample_freq: int = -1,
            normalize_advantage: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(A2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
        )

        self.actor_gradient_penalty = actor_gradient_penalty
        self.critic_gradient_penalty = critic_gradient_penalty
        self.actor_gradient_penalty_k = actor_gradient_penalty_k
        self.critic_gradient_penalty_k = critic_gradient_penalty_k

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps)

        if 'weight_decay' in self.policy_kwargs.keys():
            # This line overwrite optimizer kwargs from inherited class.
            self.policy_kwargs["optimizer_kwargs"]["weight_decay"] = policy_kwargs['weight_decay']
            del policy_kwargs['weight_decay']

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            if isinstance(self.policy.mlp_extractor, MlpExtractorWithManifoldMixup):
                values, log_prob, entropy, returns = self.policy.evaluate_actions(rollout_data.observations,
                                                                                  actions,
                                                                                  rollout_data.returns)
            else:
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                returns = rollout_data.returns
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            if self.actor_gradient_penalty > 0 or self.critic_gradient_penalty > 0:
                gradients_critic, gradients_actor = gradient_penalty_actor_critic(self.policy,
                                                                                  rollout_data.observations,
                                                                                  k_for_actor=self.actor_gradient_penalty_k,
                                                                                  k_for_critic=self.critic_gradient_penalty_k)
                gradients_critic *= self.critic_gradient_penalty
                gradients_actor *= self.actor_gradient_penalty
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + gradients_critic \
                       + gradients_actor
            else:
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())

        if self.critic_gradient_penalty > 0:
            logger.record("train/gradient_penalty_critic", gradients_critic.item())
        if self.actor_gradient_penalty > 0:
            logger.record("train/graident_penalty_actor", gradients_actor.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        if np.isinf(entropy_loss.item()) or np.isnan(entropy_loss.item()):
            raise Exception('Gradient of entropy loss KABUM!')
        if np.isinf(policy_loss.item()) or np.isnan(policy_loss.item()):
            raise Exception('Gradient of policy loss KABUM!')
        if np.isinf(value_loss.item()) or np.isnan(value_loss.item()):
            raise Exception('Gradient of value loss KABUM!')


MlpPolicy = ActorCriticPolicy

if __name__ == '__main__':
    model = A2C(ActorCriticPolicy, "CartPole-v1", verbose=1)
    model.learn(10_000)

import torch as th
from torch.nn import functional as F
from gym import spaces

from stable_baselines3 import A2C as BaseA2C
from stable_baselines3.common import logger
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.policies import ActorCriticPolicy as BaseActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.type_aliases import GymEnv
from typing import Dict, Optional, Type, Union, Any, Callable

from common.torch_layers import MlpExtractorWithDropout


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
            sde_sample_freq: int = -1,
            normalize_advantage: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            gradient_penalty: Optional[float] = None,
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

        self.gradient_penalty = gradient_penalty
        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

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

            # TODO: avoid second computation of everything because of the gradient
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            print('Policy gradient loss')
            print(advantages)
            print(logger)

            policy_loss = -(advantages * log_prob).mean() # + ... # ToDo: Dodać gradient penalty

            # Value loss using the TD(gae_lambda) target
            print('Value loss')
            print(rollout_data.returns)
            print(values)
            value_loss = F.mse_loss(rollout_data.returns, values) # + ...   # ToDo: Dodać gradient penalty

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())


MlpPolicy = ActorCriticPolicy

if __name__ == '__main__':
    model = A2C(ActorCriticPolicy, "CartPole-v1",
                policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                               'mpl_extractor_kwargs': {'dropout_rate': 0.5}},
                verbose=1)
    model.learn(10_000)

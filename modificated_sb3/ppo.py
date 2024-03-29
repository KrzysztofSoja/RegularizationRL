import numpy as np
import torch as th

from gym import spaces
from torch.nn import functional as F
from stable_baselines3.common import logger
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO as BasePPO
from stable_baselines3.common.type_aliases import GymEnv
from typing import Union, Type, Callable, Optional, Dict, Any

# ToDo: Zmienić na sys.append
try:
    from .a2c import ActorCriticPolicy
except ImportError:
    from a2c import ActorCriticPolicy
from common.torch_layers import MlpExtractorWithDropout, MlpExtractorWithManifoldMixup
from common.gradient_penalty import gradient_penalty_actor_critic


__all__ = ['PPO', 'MlpPolicy']


class PPO(BasePPO):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        actor_gradient_penalty: float = 0.0,
        critic_gradient_penalty: float = 0.0,
        actor_gradient_penalty_k: float = 1.0,
        critic_gradient_penalty_k: float = 1.0,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        policy_kwargs = dict() if policy_kwargs is None else policy_kwargs

        if "optimizer_class" not in policy_kwargs:
            policy_kwargs["optimizer_class"] = th.optim.Adam
            policy_kwargs["optimizer_kwargs"] = dict(eps=1e-5)

        if 'weight_decay' in policy_kwargs.keys():
            policy_kwargs["optimizer_kwargs"]["weight_decay"] = policy_kwargs['weight_decay']
            del policy_kwargs['weight_decay']

        self.actor_gradient_penalty = actor_gradient_penalty
        self.critic_gradient_penalty = critic_gradient_penalty
        self.actor_gradient_penalty_k = actor_gradient_penalty_k
        self.critic_gradient_penalty_k = critic_gradient_penalty_k

        super(PPO, self).__init__(
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

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                if isinstance(self.policy.mlp_extractor, MlpExtractorWithManifoldMixup):
                    values, log_prob, entropy, returns = self.policy.evaluate_actions(rollout_data.observations,
                                                                                      actions,
                                                                                      rollout_data.returns)
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    returns = rollout_data.returns
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                if self.actor_gradient_penalty > 0 or self.critic_gradient_penalty > 0:
                    gradients_critic, gradients_actor = gradient_penalty_actor_critic(self.policy, rollout_data.observations,
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
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        temp_entropy_loss = np.mean(entropy_losses)
        temp_pg_loss = np.mean(pg_losses)
        temp_value_loss = np.mean(value_losses)

        logger.record("train/entropy_loss", temp_entropy_loss)
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

        if self.critic_gradient_penalty > 0:
            logger.record("train/gradient_penalty_critic", gradients_critic.item())
        if self.actor_gradient_penalty > 0:
            logger.record("train/gradient_penalty_actor", gradients_actor.item())

        if np.isinf(temp_entropy_loss) or np.isnan(temp_entropy_loss):
            raise Exception('Gradient KABUM! entropy_loss.')
        if np.isinf(temp_pg_loss) or np.isnan(temp_pg_loss):
            raise Exception('Gradient KABUM! policy_gradient_loss.')
        if np.isinf(temp_value_loss) or np.isnan(temp_value_loss):
            raise Exception('Gradient KABUM! value_loss.')


MlpPolicy = ActorCriticPolicy


if __name__ == '__main__':
    model = PPO(MlpPolicy, "CartPole-v1",
                policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                               'mpl_extractor_kwargs': {'dropout_rate': 0.5},
                               'weight_decay': 0.1},
                verbose=1)
    model.learn(10_000)

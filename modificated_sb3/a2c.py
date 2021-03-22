import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3 import A2C as BaseA2C
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.policies import ActorCriticPolicy as BaseActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.utils import explained_variance
from typing import Dict, Optional, Type, Union, Any, Callable
from gym import spaces


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

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps)

        if 'weight_decay' in self.policy_kwargs.keys():
            self.policy_kwargs["optimizer_kwargs"]["weight_decay"] = policy_kwargs['weight_decay']
            del policy_kwargs['weight_decay']

        if _init_setup_model:
            self._setup_model()


MlpPolicy = ActorCriticPolicy

if __name__ == '__main__':
    model = A2C(ActorCriticPolicy, "CartPole-v1",
                policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                               'mpl_extractor_kwargs': {'dropout_rate': 0.5},
                               'weight_decay': 0.001},
                ent_coef=0.1,
                verbose=1)
    model.learn(10_000)

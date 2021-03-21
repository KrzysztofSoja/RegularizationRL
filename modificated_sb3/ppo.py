import torch as th

from stable_baselines3 import PPO as BasePPO
from stable_baselines3.common.type_aliases import GymEnv
from typing import Union, Type, Callable, Optional, Dict, Any

try:
    from .a2c import ActorCriticPolicy
except ImportError:
    from a2c import ActorCriticPolicy
from common.torch_layers import MlpExtractorWithDropout


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

        if "optimizer_class" not in policy_kwargs:
            policy_kwargs["optimizer_class"] = th.optim.Adam
            policy_kwargs["optimizer_kwargs"] = dict(eps=1e-5)

        if 'weight_decay' in policy_kwargs.keys():
            policy_kwargs["optimizer_kwargs"]["weight_decay"] = policy_kwargs['weight_decay']
            del policy_kwargs['weight_decay']

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


MlpPolicy = ActorCriticPolicy


if __name__ == '__main__':
    model = PPO(MlpPolicy, "CartPole-v1",
                policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                               'mpl_extractor_kwargs': {'dropout_rate': 0.5},
                               'weight_decay': 0.1},
                verbose=1)
    model.learn(10_000)

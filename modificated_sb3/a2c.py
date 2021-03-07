from stable_baselines3.common.policies import ActorCriticPolicy as BaseActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from typing import Dict, Optional, Type, Union, Any

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


MlpPolicy = ActorCriticPolicy

if __name__ == '__main__':
    model = A2C(ActorCriticPolicy, "CartPole-v1",
                policy_kwargs={'mlp_extractor_class': MlpExtractorWithDropout,
                               'mpl_extractor_kwargs': {'dropout_rate': 0.5}},
                verbose=1)
    model.learn(10_000)

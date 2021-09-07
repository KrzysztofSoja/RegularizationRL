import torch as th
import torch.nn as nn

from typing import Tuple


def gradient_penalty_actor_critic(model: nn.Module, data: th.Tensor,
                                  k_for_critic: float = 1,
                                  k_for_actor: float = 1) -> Tuple[th.Tensor, th.Tensor]:
    data = th.autograd.Variable(data, requires_grad=True)
    prediction = model(data)

    gradients_critics = th.autograd.grad(outputs=prediction[1], inputs=data,
                                         grad_outputs=th.ones(prediction[1].size()),
                                         create_graph=True, retain_graph=True)[0]
    gradients_actor = th.autograd.grad(outputs=prediction[2], inputs=data,
                                       grad_outputs=th.ones(prediction[2].size()),
                                       create_graph=True, retain_graph=True)[0]
    gradients_critics, gradients_actor = gradients_critics.to(th.float), gradients_actor.to(th.float)

    gradients_critics = gradients_critics.view(data.shape[0], -1)
    gradients_critics_norm = th.sqrt(th.sum(gradients_critics ** 2, dim=1) + 1e-12)

    gradients_actor = gradients_actor.view(data.shape[0], -1)
    gradients_actor_norm = th.sqrt(th.sum(gradients_actor ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_critics_norm - k_for_critic) ** 2).mean(), ((gradients_actor_norm - k_for_actor) ** 2).mean()


def gradient_penalty(model: nn.Module, data: th.Tensor, k: float = 1,) -> th.Tensor:
    data = th.autograd.Variable(data, requires_grad=True)
    prediction = model(data)

    gradients = th.autograd.grad(outputs=prediction[1], inputs=data,
                                 grad_outputs=th.ones(prediction[1].size()),
                                 create_graph=True, retain_graph=True)[0]
    gradients = gradients.to(th.float)

    gradients = gradients.view(data.shape[0], -1)
    gradients_norm = th.sqrt(th.sum(gradients ** 2, dim=1) + 1e-12)

    return ((gradients_norm - k) ** 2).mean()


def gradient_penalty_for_continues_critic(network: nn.Module,
                                          features_extractor: nn.Module,
                                          observations: th.Tensor,
                                          actions: th.Tensor,
                                          k: float = 1,
                                          share_feature_extractor: bool = False) -> th.Tensor:
    with th.set_grad_enabled(not share_feature_extractor):
        features = features_extractor(observations)
    qvalue_input = th.cat([features, actions], dim=1)
    return gradient_penalty(network, qvalue_input, k)

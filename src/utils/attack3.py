# Attack 3 Implementation.
# Used for Make-Head-Disappear Experiment.

import torch

import numpy as np

from src.utils.helpers import find_high_activation_crop, get_all_xy


def similarity_score(ppnet_multi, ppnet, preprocess_fn, x, pid=1):
    """
    Forward propagate through ProtoPNet and return the required objective function to minimize.
    """
    x = preprocess_fn(x.squeeze(0)).unsqueeze(0)
    logits, min_distances = ppnet_multi(x)
    conv_output, distances = ppnet.push_forward(x)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == "linear":
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    idx = 0
    return prototype_activations[idx][pid]


def pgd(
    x,
    mask,
    pid,
    net_multi,
    net,
    preprocess_fn,
    attack_steps=40,
    attack_lr=2 / 255,
    attack_eps=8 / 255,
    random_init=True,
    minimize=True,
    clip_min=0.0,
    clip_max=1.0,
):
    """
    Perform PGD to minimize similarty of a prototype wrt to given image patch.
    """
    x_adv = x.clone()
    if random_init:
        x_adv = torch.clamp(
            x_adv + torch.empty_like(x).uniform_(-attack_eps, attack_eps),
            clip_min,
            clip_max,
        )

    for i in range(attack_steps):
        x_adv.requires_grad = True

        net_multi.zero_grad()
        sim_score = similarity_score(net_multi, net, preprocess_fn, x_adv, pid)

        loss = -sim_score
        loss.backward(retain_graph=True)
        grad = x_adv.grad.detach()
        grad = grad.sign()
        if minimize:
            x_adv = x_adv + attack_lr * grad * mask
        else:
            x_adv = x_adv - attack_lr * grad * mask

        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps) * mask
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, clip_min, clip_max)

    r_adv = x_adv - x
    sim_score = similarity_score(net_multi, net, preprocess_fn, x_adv, pid)
    return x_adv, r_adv, sim_score.item()

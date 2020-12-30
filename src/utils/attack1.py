import torch

import numpy as np
import itertools
import cv2

from src.utils.helpers import find_high_activation_crop, get_all_xy


def similarity_score(ppnet_multi, ppnet, preprocess_fn, x, pid=1, loc=None, act_loc=None):
    assert loc != None, 'please provide the locations in feature map as a list of dict (x, y)'
    mask, loc = get_all_xy(loc)
    
    x = preprocess_fn(x.squeeze(0)).unsqueeze(0)
    logits, min_distances = ppnet_multi(x)
    conv_output, distances = ppnet.push_forward(x)
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    idx = 0
    sim = torch.zeros(len(loc)).cuda()
    p_act = prototype_activation_patterns[idx][pid].clone()
    for i, l in enumerate(loc):
        sim[i] = prototype_activation_patterns[idx][pid][l[0], l[1]]
        p_act[l[0], l[1]] = 0
    local_max_sim = torch.max(sim)
    global_max_sim = torch.max(prototype_activation_patterns[idx][pid])
    
    if act_loc is not None:
        act_loc = [yx for yx in itertools.product(np.arange(act_loc[0], act_loc[1]), 
                                                  np.arange(act_loc[2], act_loc[3]))]
        act_sim = torch.zeros(len(act_loc)).cuda()
        for i, l in enumerate(act_loc):
            act_sim[i] = p_act[l[0], l[1]]
        return torch.mean(sim) - torch.mean(act_sim), local_max_sim, global_max_sim, prototype_activation_patterns
    else:
        return torch.mean(sim), local_max_sim, global_max_sim, prototype_activation_patterns[idx]


def pgd(x, mask, pid, loc, net_multi, net, preprocess_fn, attack_steps=40, attack_lr=2/255, attack_eps=8/255, 
        random_init=True, minimize=False, clip_min=0.0, clip_max=1.0, idx=0, grid=7):
    
    img_size = net_multi.module.img_size
    x_adv = x.clone()
    if random_init:
        x_adv = torch.clamp(x_adv + torch.empty_like(x).uniform_(-attack_eps, attack_eps), clip_min, clip_max) 

    sim_score, _, _, prototype_activation_patterns = similarity_score(net_multi, net, preprocess_fn, x, 
                                                                      pid, loc=loc)
    activation_pattern = prototype_activation_patterns[pid].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size), 
                                              interpolation=cv2.INTER_CUBIC)
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
    step = img_size//grid
    act_loc = [high_act_patch_indices[0]//step, high_act_patch_indices[1]//step, 
               high_act_patch_indices[2]//step, high_act_patch_indices[3]//step]
    
    for i in range(attack_steps):
        x_adv.requires_grad = True

        net_multi.zero_grad()
        sim_score, _, _, _ = similarity_score(net_multi, net, preprocess_fn, x_adv, pid, 
                                              loc=loc, act_loc=act_loc)

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
    sim_score, _, _, _ = similarity_score(net_multi, net, preprocess_fn, x_adv, pid, loc)
    return x_adv, r_adv, sim_score.item()
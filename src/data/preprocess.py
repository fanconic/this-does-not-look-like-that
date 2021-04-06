# Helper functions for preprocessing images.

import torch


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

tensor_mean = torch.tensor(mean).view(3, 1, 1).cuda()
tensor_std = torch.tensor(std).view(3, 1, 1).cuda()

upper_limit = (1 - tensor_mean) / tensor_std
lower_limit = (0 - tensor_mean) / tensor_std


def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    """
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    """
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    """
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    """
    return undo_preprocess(x, mean=mean, std=std)

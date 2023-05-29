# Helper functions.

import os
import itertools
import torch
import numpy as np
from matplotlib import pyplot as plt

import copy
from src.data.preprocess import undo_preprocess_input_function


def list_of_distances(X, Y):
    return torch.sum(
        (torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1
    )


def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.0)


def makedir(path):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + "\n")


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def visualize_image_grid(preprocess_fn=None, images=None, titles=None, ncols=3):
    if titles:
        assert len(titles) == ncols
    N = np.ceil(len(images) / ncols).astype(int)
    plt.figure(figsize=(3 * ncols, 3 * N))
    for i, image in enumerate(images):
        plt.subplot(N, ncols, i + 1)
        if preprocess_fn:
            plt.imshow(preprocess_fn(image))
        else:
            plt.imshow(image)
        if i < ncols and titles:
            plt.title(titles[i])
        plt.axis("off")


def plot_grid_on_image(img_variable, img_size, grid=7):
    img = torch2numpy(undo_preprocess_input_function(img_variable))
    assert img_size == img.shape[0] and img_size == img.shape[1], "shapes must match."
    step_x, step_y = img.shape[0] // grid, img.shape[1] // grid
    img[:, ::step_y, :] = (0, 0, 0)
    img[::step_x, :, :] = (0, 0, 0)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.xticks(np.arange(0, img.shape[1], step_y), np.arange(0, grid, 1))
    plt.yticks(np.arange(0, img.shape[0], step_x), np.arange(0, grid, 1))
    plt.title("Test input image. With {}x{} grid.".format(grid, grid))
    plt.show()


def torch2numpy(imgs, index=0):
    img_copy = copy.deepcopy(imgs[index : index + 1])
    img_copy = img_copy[0]
    np_img = img_copy.detach().cpu().numpy()
    np_img = np.transpose(np_img, [1, 2, 0])
    return np_img


def get_all_xy(loc, grid=7):
    xs, ys = [], []
    for l in loc:
        xs.append(l[1])
        ys.append(l[0])
    xmin, ymin = np.min(xs), np.min(ys)
    xmax, ymax = np.max(xs) + 1, np.max(ys) + 1
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, grid), min(ymax, grid)
    return (
        [ymin, ymax, xmin, xmax],
        [yx for yx in itertools.product(np.arange(ymin, ymax), np.arange(xmin, xmax))],
    )


def get_image_patch_position(loc, img_size, grid=7):
    xs, ys = [], []
    for l in loc:
        xs.append(l[1])
        ys.append(l[0])
    xmin, ymin = np.min(xs), np.min(ys)
    xmax, ymax = np.max(xs) + 1, np.max(ys) + 1
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, grid), min(ymax, grid)
    xgrid = np.arange(0, img_size + 1, img_size // grid)
    ygrid = np.arange(0, img_size + 1, img_size // grid)
    return [ygrid[ymin], ygrid[ymax], xgrid[xmin], xgrid[xmax]]

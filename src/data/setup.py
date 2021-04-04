# Setup the training and testing data.

import os
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

from tqdm import tqdm
from settings import colab, num_classes, username


def setup_test_image(idx):
    """
    Setup a single test image for inference.
    Args:
        idx (int): Index of the test image.
    """
    if colab:
        DIR = "/content/"
        OUT = "/content/PPNet/"
    else:
        DIR = "/cluster/scratch/{}/PPNet/".format(username)
        OUT = "/scratch/PPNet/"

    train_test_split = pd.read_csv(
        DIR + "CUB_200_2011/train_test_split.txt", sep=" ", header=None
    )
    train_test_split.columns = ["image_id", "is_train"]
    train_test_split.set_index("image_id", inplace=True)

    images = pd.read_csv(DIR + "CUB_200_2011/images.txt", sep=" ", header=None)
    images.columns = ["image_id", "file_name"]
    images.set_index("image_id", inplace=True)
    bb = pd.read_csv(DIR + "CUB_200_2011/bounding_boxes.txt", sep=" ", header=None)
    bb.columns = ["image_id", "x", "y", "width", "height"]
    bb.set_index("image_id", inplace=True)

    if not os.path.isdir(OUT):
        os.mkdir(OUT)
        os.mkdir(OUT + "datasets")
        os.mkdir(OUT + "datasets/cub200_cropped")
        os.mkdir(OUT + "datasets/cub200_cropped/train_cropped")
        os.mkdir(OUT + "datasets/cub200_cropped/test_cropped")

    image_id = idx
    file_name = images.loc[idx, "file_name"]

    is_train = train_test_split.loc[idx, "is_train"]
    typ = "train" if is_train else "test"
    if not os.path.isdir(
        OUT + "datasets/cub200_cropped/" + typ + "_cropped/" + file_name.split("/")[0]
    ):
        os.mkdir(
            OUT
            + "datasets/cub200_cropped/"
            + typ
            + "_cropped/"
            + file_name.split("/")[0]
        )

    x, y, width, height = bb.loc[idx]
    image = cv2.imread(DIR + "CUB_200_2011/images/" + file_name)
    image = image[int(y) : int(y + height), int(x) : int(x + width)]
    print("Image id: {}.\tIsTrain: {}".format(idx, is_train))
    print("File name: {}".format(file_name))

    out_file = OUT + "datasets/cub200_cropped/" + typ + "_cropped/" + file_name
    cv2.imwrite(out_file, image)


def setup_data(num_classes=200):
    """
    Setup training and testing data.
    Args:
        num_classes (int): Number of classes to use (max: 200).
    """
    print("No. of classes used: ", num_classes)
    if colab:
        DIR = "/content/"
        OUT = "/content/PPNet/"
    else:
        DIR = "/cluster/scratch/{}/PPNet/".format(username)
        OUT = "/scratch/PPNet/"

    train_test_split = pd.read_csv(
        DIR + "CUB_200_2011/train_test_split.txt", sep=" ", header=None
    )
    train_test_split.columns = ["image_id", "is_train"]
    train_test_split.head()

    images = pd.read_csv(DIR + "CUB_200_2011/images.txt", sep=" ", header=None)
    images.columns = ["image_id", "file_name"]
    images.head()

    bb = pd.read_csv(DIR + "CUB_200_2011/bounding_boxes.txt", sep=" ", header=None)
    bb.columns = ["image_id", "x", "y", "width", "height"]
    bb.head()

    assert images.shape[0] == train_test_split.shape[0]
    assert images.shape[0] == bb.shape[0]

    if not os.path.isdir(OUT):
        os.mkdir(OUT)
        os.mkdir(OUT + "datasets")
        os.mkdir(OUT + "datasets/cub200_cropped")
        os.mkdir(OUT + "datasets/cub200_cropped/train_cropped")
        os.mkdir(OUT + "datasets/cub200_cropped/test_cropped")

    for i in tqdm(images.index):
        image_id = images.loc[i, "image_id"]
        file_name = images.loc[i, "file_name"]

        if int(file_name.split(".")[0]) > num_classes:
            continue

        is_train = train_test_split.loc[i, "is_train"]
        assert (
            image_id == bb.loc[i, "image_id"]
            and image_id == train_test_split.loc[i, "image_id"]
        )
        typ = "train" if is_train else "test"
        if not os.path.isdir(
            OUT
            + "datasets/cub200_cropped/"
            + typ
            + "_cropped/"
            + file_name.split("/")[0]
        ):
            os.mkdir(
                OUT
                + "datasets/cub200_cropped/"
                + typ
                + "_cropped/"
                + file_name.split("/")[0]
            )

        _, x, y, width, height = bb.loc[i]
        image = cv2.imread(DIR + "CUB_200_2011/images/" + file_name)
        image = image[int(y) : int(y + height), int(x) : int(x + width)]

        out_file = OUT + "datasets/cub200_cropped/" + typ + "_cropped/" + file_name
        cv2.imwrite(out_file, image)


if __name__ == "__main__":
    setup_data(num_classes=num_classes)

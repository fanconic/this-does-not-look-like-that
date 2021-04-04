# Setup the training and testing data.

import os
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

from tqdm import tqdm
from settings import colab, num_classes, username, JPEG_QUALITY



def setup_compress(num_classes=200):
    """
    Setup training and testing data.
    Args:
        num_classes (int): Number of classes to use (max: 200).
    """
    print ('No. of classes used: ', num_classes)
    if colab:
        DIR = '/content/'
        OUT = '/content/PPNet/'
    else:
        DIR = '/cluster/scratch/{}/PPNet/'.format(username)
        OUT = '/scratch/PPNet/'

    train_test_split = pd.read_csv(DIR+'CUB_200_2011/train_test_split.txt', sep=' ', header=None)
    train_test_split.columns = ['image_id', 'is_train']
    train_test_split.head()

    images = pd.read_csv(DIR+'CUB_200_2011/images.txt', sep=' ', header=None)
    images.columns = ['image_id', 'file_name']
    images.head()


    assert images.shape[0] == train_test_split.shape[0]

    if not os.path.isdir(OUT):
        os.mkdir(OUT)
        os.mkdir(OUT+'datasets')
        os.mkdir(OUT+'datasets/cub200_cropped')
        os.mkdir(OUT+'datasets/cub200_cropped/train_cropped')
        os.mkdir(OUT+'datasets/cub200_cropped/test_cropped')

    for i in tqdm(images.index):
        image_id = images.loc[i, 'image_id']
        file_name = images.loc[i, 'file_name']
        
        if int(file_name.split('.')[0]) > num_classes:
            continue
        
        is_train = train_test_split.loc[i, 'is_train']
        if is_train:
            if not os.path.isdir(OUT+'datasets/cub200_cropped/train_cropped/' + file_name.split('/')[0]) :
                os.mkdir(OUT+'datasets/cub200_cropped/train_cropped/' + file_name.split('/')[0])

    
            image = cv2.imread(DIR+'CUB_200_2011/images/'+file_name)

            out_file = OUT + 'datasets/cub200_cropped/train_cropped/' + file_name.split('/')[0] + '/' + 'compressed_' + file_name.split('/')[1]
            cv2.imwrite(out_file, image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


if __name__ == '__main__':
    setup_compress(num_classes=num_classes)

# Perform offline data augmentation for training data.

import Augmentor

import os
import sys
sys.path.insert(0, '.')

from settings import data_path, username


def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__


def makedir(path):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def augment():        
    """
    Perform 40x data augmentation for each training image.
    """
    datasets_root_dir = data_path
    dir = datasets_root_dir + 'train_cropped/'
    target_dir = datasets_root_dir + 'train_cropped_augmented/'

    makedir(target_dir)
    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]
    
    disable_print()
    
    for i in range(len(folders)):
        fd = folders[i]
        tfd = target_folders[i]
        # rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.skew(probability=1, magnitude=0.2)  # max 45 degrees
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
        # random_distortion
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            p.process()
        del p
    
    enable_print()
    

if __name__ == '__main__':
    augment()
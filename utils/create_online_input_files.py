
import os
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import os.path as osp
import pickle
from collections import Counter
from random import seed, choice, sample
from collections import Counter


def create_input_files(image_folder,split, output_folder):
    image_folder=os.path.join(image_folder,split)
    test_image_paths = os.listdir(image_folder)
    test_image_id = [str(int(one.split('_')[-1][:-4])) +
                     '.jpg' for one in test_image_paths]

    seed(123)

    with h5py.File(os.path.join(output_folder, 'online_' + split + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        
        # Create dataset inside HDF5 file to store images
        images = h.create_dataset(
            'images', (len(test_image_paths), 3, 256, 256), dtype='uint8')

        print("\nReading test images, storing to file...\n")

        for i, path in enumerate(tqdm(test_image_paths)):

            # Read images H,W,C ,灰度扩充到3维，每维一样
            # 可修改为其他的读取方式
            img = imread(os.path.join(image_folder,path))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = imresize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            images[i] = img

    with open(os.path.join(output_folder, 'online_' + split + '_ids' + '.json'), 'w') as j:
        json.dump(test_image_id, j)

if __name__ == '__main__':
    split='val2014'
    image_folder = '/home/lkk/datasets/coco2014/'
    output_folder = '/home/lkk/datasets/coco2014/'
    create_input_files(image_folder, split, output_folder)

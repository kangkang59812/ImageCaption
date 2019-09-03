
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


# def create_image_attributes(karpathy_json_path, attributes_vocab_path):
#     with open(karpathy_json_path, 'r') as j:
#         data = json.load(j)
#     with open(attributes_vocab_path, 'r') as j:
#         word_map = json.load(j)['word_map']
#     k = word_map.keys()
#     img_tag = dict()
#     closed_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'at', 'are', 'next',
#                     'that', 'it']
#     for img in data['images']:
#         img_tag[img['imgid']] = list()
#         for c in img['sentences']:
#             bb = [word for word in c['tokens'] if word not in closed_words]
#             # print(bb)
#             bd = [word for word in bb if word in k]
#             if len(bd) == 0:
#                 print(bd)
#                 print(img)
#                 a = 1
#                 d = img
#                 break
#             img_tag[img['imgid']].append(bb)

#     img_tag[img['imgid']] = img_tag[img['imgid']]


def create_attributes_vocab(karpathy_json_path, top=1039):
    # only choose top-1024 except 'a,on...'
    # read caption file，本身就小写处理过了
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    #
    word_freq = Counter()
    for img in data['images']:
        for c in img['sentences']:
            word_freq.update(c['tokens'])
    order = sorted(word_freq.items(),
                   key=lambda item: item[1], reverse=True)[:top]
    closed_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'at', 'are', 'next',
                    'that', 'it']
    # create word-to-id and id-to-word
    words = [w[0] for w in order if w[0] not in closed_words]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    map_word = {v+1: k for v, k in enumerate(words)}

    return word_map, map_word, word_freq


def create_input_files(dataset, karpathy_json_path, image_folder, features_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=50):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco2014', 'flickr8k', 'flickr30k', 'coco2017'}
    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    with open(os.path.join(features_folder, 'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)

    with open(os.path.join(features_folder, 'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []

    train_image_det = []
    val_image_det = []
    test_image_det = []

    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if 'coco2014' in dataset else os.path.join(
            image_folder, img['filename'])
        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t", train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v", val_data[image_id]))
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            if image_id in val_data:
                val_image_det.append(("v", val_data[image_id]))

        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            if image_id in val_data:
                test_image_det.append(("v", val_data[image_id]))

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)
    assert len(train_image_det) == len(train_image_captions)
    assert len(val_image_det) == len(val_image_captions)
    assert len(test_image_det) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + \
        str(captions_per_image) + '_cap_per_img_' + \
        str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, impaths2, imcaps, split in [(train_image_paths, train_image_det, train_image_captions, 'TRAIN'),
                                             (val_image_paths, val_image_det,
                                              val_image_captions, 'VAL'),
                                             (test_image_paths, test_image_det, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset(
                'images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions 随机重复，补足5个;多了随机挑5个
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i])
                                            for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images H,W,C ,灰度扩充到3维，每维一样
                # 可修改为其他的读取方式
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths  不算word_map['<pad>']
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * \
                captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

    # Save bottom up features indexing to JSON files
    with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_image_det, j)

    with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_image_det, j)

    with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_image_det, j)


def create_flickr8k(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                    max_len=50):

    with open('/home/lkk/datasets/flickr8k/dataset.json', 'r') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []

    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            for one in c['tokens']:
                if str.isalpha(one):
                    word_freq.update([one])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if 'coco2014' in dataset else os.path.join(
            image_folder, img['filename'])
        image_id = img['imgid']

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    base_filename = dataset + '_' + \
        str(captions_per_image) + '_cap_per_img_' + \
        str(min_word_freq) + '_min_word_freq'

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                             (val_image_paths, val_image_captions, 'VAL'),
                                             (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset(
                'images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions 随机重复，补足5个;多了随机挑5个
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i])
                                            for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images H,W,C ,灰度扩充到3维，每维一样
                # 可修改为其他的读取方式
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths  不算word_map['<pad>']
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * \
                captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':
    # Create input files (along with word map)
    # coco2014 and coco2017
    # create_input_files(dataset='coco2014',
    #                    karpathy_json_path='/home/lkk/datasets/coco2014/dataset_coco.json',
    #                    image_folder='/home/lkk/datasets/coco2014/',
    #                    captions_per_image=5,
    #                    min_word_freq=5,
    #                    output_folder='/home/lkk/datasets/cocotest/',
    #                    features_folder='/home/lkk/dataset/',
    #                    max_len=50)
    create_flickr8k(dataset='flickr',
                    karpathy_json_path='/home/lkk/datasets/flickr8k/dataset.json',
                    image_folder='/home/lkk/datasets/flickr8k/Flicker8k_Dataset',
                    captions_per_image=5,
                    min_word_freq=5,
                    output_folder='/home/lkk/datasets/flickr8k/',
                    max_len=50)
    # create_image_attributes(karpathy_json_path='/home/lkk/datasets/coco2014/dataset_coco.json',
    #                         attributes_vocab_path='/home/lkk/datasets/coco2014/attributes_map.json')

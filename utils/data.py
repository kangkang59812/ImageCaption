
import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from torch import nn
from collections import OrderedDict


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, tag_flag=False, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.tag_flag = tag_flag
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # word map
        with open(os.path.join(data_folder, 'WORDMAP_coco_5_cap_per_img_5_min_word_freq' + '.json'), 'r') as j:
            self.word_map = json.load(j)
            self.rev_word_map = {v: k for k, v in self.word_map.items()}
        # 加载attributes map
        with open(os.path.join(data_folder, 'attributes_map' + '.json'), 'r') as j:
            self.attributes_map = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.tag_flag:
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            all_caplens = torch.LongTensor(self.caplens[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            tags = set()
            for c, l in zip(all_captions, all_caplens):
                for j in range(1, l.item()):
                    if self.rev_word_map[c[j].item()] in self.attributes_map['word_map'].keys():
                        tags.add(self.attributes_map['word_map'][self.rev_word_map[c[j].item()]])

            tags_target = torch.zeros(len(self.attributes_map['word_map']))
            tags_target[list(map(lambda n:n-1, list(tags)))]=1
            tags_target = torch.Tensor(tags_target)

        if self.split is 'TRAIN':
            if self.tag_flag:
                return img, caption, caplen, tags_target
            else:
                return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            # all_captions = torch.LongTensor(
            #     self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            if self.tag_flag:
                return img, caption, caplen, all_captions, tags_target
            else:
                return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class fCaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.train_hf = h5py.File(data_folder + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(data_folder + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        # Captions per image
        self.cpi = 5

        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):

        # The Nth caption corresponds to the (N // captions_per_image)th image
        objdet = self.objdet[i // self.cpi]

        # Load bottom up image features
        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class BothCaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, features_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.train_hf = h5py.File(features_folder + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(features_folder + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # 2
        # Load encoded captions
        with open(os.path.join(features_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions2 = json.load(j)

        # Load caption lengths
        with open(os.path.join(features_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens2 = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(features_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        objdet = self.objdet[i // self.cpi]

        # Load bottom up image features
        if objdet[0] == "v":
            img2 = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img2 = torch.FloatTensor(self.train_features[objdet[1]])

        caption2 = torch.LongTensor(self.captions[i])

        caplens2 = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    data_folder = '/home/lkk/datasets/coco2014'
    features_folder = '/home/lkk/dataset/'
    data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    dataset = CaptionDataset(
        data_folder, data_name, 'TRAIN')

    # 每5个同一张图，不同captions
    # [36,2048]的features，52长度的caption，里面是单词的索引，根据长度多余的补0， caption实际长度
    data = dataset[10]
    print('')

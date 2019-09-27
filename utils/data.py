
import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from torch import nn
from collections import OrderedDict
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

attributes = ['people', 'airplane', 'children', 'bag', 'bear', 'bike', 'boat', 'book', 'bottle', 'bus', 'car', 'cat', 'computer', 'dog', 'doughnut', 'drink', 'fridge', 'giraffe', 'glass', 'horse', 'image', 'cellphone', 'monitor ', 'motorcycle', 'mountain', 'play', 'road', 'sink', 'sit', 'skate', 'watching', 'skiing', 'stand', 'snow', 'snowboard', 'table', 'swing', 'television', 'tree', 'wood', 'zebra', 'air', 'airport', 'apple', 'ball', 'banana', 'baseball', 'basket', 'bat', 'bathroom', 'bathtub', 'beach', 'bedroom', 'bed', 'beer', 'bench', 'big', 'bird', 'black', 'blanket', 'blue', 'board', 'bowl', 'box', 'bread', 'bridge', 'broccoli', 'brown', 'building', 'cabinet', 'cake', 'camera', 'candle', 'carrot', 'carry', 'catch', 'chair', 'cheese', 'chicken', 'chocolate', 'christmas', 'church', 'city', 'clock', 'cloud', 'coat', 'coffee', 'couch', 'court', 'cow', 'cup', 'cut', 'decker', 'desk', 'dining', 'dirty', 'dish', 'door', 'drive', 'eat', 'elephant', 'face', 'fancy', 'fence', 'field', 'fire', 'fish', 'flag', 'flower', 'fly', 'food', 'forest', 'fork', 'frisbee', 'fruit', 'furniture', 'giant', 'graffiti', 'grass', 'gray', 'green', 'ground', 'group', 'hair', 'hand', 'hat',
              'head', 'helmet', 'hill', 'hit', 'hold', 'hotdog', 'house', 'hydrant', 'ice', 'island', 'jacket', 'jump', 'keyboard', 'kitchen', 'kite', 'kitten', 'knife', 'lake', 'laptop', 'large', 'lay', 'lettuce', 'light', 'lit', 'little', 'look', 'luggage', 'lush', 'market', 'meat', 'metal', 'microwave', 'mouse', 'mouth', 'ocean', 'office', 'old', 'onion', 'orange', 'oven', 'palm', 'pan', 'pant', 'paper', 'park', 'pen', 'pillow', 'pink', 'pizza', 'plant', 'plastic', 'plate', 'player', 'police', 'potato', 'pull', 'purple', 'race', 'racket', 'racquet', 'rail', 'rain', 'read', 'red', 'restaurant', 'ride', 'river', 'rock', 'room', 'run', 'salad', 'sand', 'sandwich', 'sea', 'seat', 'sheep', 'shelf', 'ship', 'shirt', 'shorts', 'shower', 'sign', 'silver', 'sky', 'sleep', 'small', 'smiling', 'sofa', 'station', 'stone', 'suit', 'suitcase', 'sunglasses', 'surfboard', 'surfing', 'swim', 'take', 'talk', 'tall', 'teddy', 'tennis', 'through', 'toddler', 'toilet', 'tomato', 'top', 'towel', 'tower', 'toy', 'track', 'traffic', 'train', 'truck', 'two', 'umbrella', 'vegetable', 'vehicle', 'wait', 'walk', 'wall', 'water', 'wear', 'wedding', 'white', 'wii', 'window', 'wine', 'yellow', 'young', 'zoo']


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
        with open(os.path.join(data_folder, 'WORDMAP_'+data_name + '.json'), 'r') as j:
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

        all_captions = torch.LongTensor(self.captions[(
            (i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        all_caplens = torch.LongTensor(self.caplens[(
            (i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

        if self.tag_flag:
            tags = set()
            for c, l in zip(all_captions, all_caplens):
                for j in range(1, l.item()):
                    if self.rev_word_map[c[j].item()] in self.attributes_map['word_map'].keys():
                        tags.add(
                            self.attributes_map['word_map'][self.rev_word_map[c[j].item()]])

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


class OnlineCaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, transform=None):

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(
            data_folder, 'online_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # 加载ids
        with open(os.path.join(data_folder, 'online_' + data_name + '_ids' + '.json'), 'r') as j:
            self.ids = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.ids)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.ids[i][:-4]

    def __len__(self):
        return self.dataset_size


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


class CaptionDataset256(Dataset):
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

        all_captions = torch.LongTensor(self.captions[(
            (i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        all_caplens = torch.LongTensor(self.caplens[(
            (i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])

        if self.tag_flag:
            tags = set()
            for c, l in zip(all_captions, all_caplens):
                for j in range(1, l.item()-1):
                    # if self.rev_word_map[c[j].item()] in self.attributes_map['word_map'].keys():
                    tags.add(self.rev_word_map[c[j].item()])
            tagged_sent = pos_tag(list(tags))   # 获取单词词性
            wnl = WordNetLemmatizer()
            lemmas_sent = set()
            for tag in tagged_sent:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                lemmas_sent.add(wnl.lemmatize(
                    tag[0], pos=wordnet_pos))  # 词形还原

            tokens = [attributes.index(one)
                      for one in lemmas_sent if one in attributes]
            tags_target = torch.zeros(len(attributes))
            tags_target[list(map(lambda n: n, tokens))]=1
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


if __name__ == "__main__":
    data_folder = '/home/lkk/datasets/coco2014'
    # features_folder = '/home/lkk/dataset/'
    # data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    # dataset = CaptionDataset(
    #     data_folder, data_name, 'TRAIN', True)

    # # 每5个同一张图，不同captions
    # # [36,2048]的features，52长度的caption，里面是单词的索引，根据长度多余的补0， caption实际长度
    # data = dataset[10]
    # print('')
    data_name = 'test2014'
    test_loader = OnlineCaptionDataset(data_folder, data_name)
    data = test_loader[0]
    print('')

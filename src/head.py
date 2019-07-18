import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict


class Head(nn.Module):
    '''
    use res101's structure before block2
    '''
    def __init__(self, model='res101', freeze=True):

        super(Head, self).__init__()
        if model == 'res101':
            base_model = torchvision.models.resnet101(
                pretrained=True)
            self.base_model = torch.nn.Sequential(OrderedDict([
                ('conv1', base_model.conv1),
                ('bn1', base_model.bn1),
                ('relu', base_model.relu),
                ('maxpool', base_model.maxpool),
                ('layer1', base_model.layer1),
            ]))

        if freeze:
            self.freeze()

    def forward(self, x):

        out = self.base_model(x)

        return out

    def freeze(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

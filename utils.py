import os
import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

def net_require_grad(nets, requires_grad=False):
    for net in nets:
        for params in net.parameters():
            params.requires_grad = requires_grad

class Sample_buffer(object):
    def __init__(self, history_size=50): # in the paper it says 50
        self.history_size = history_size
        self.current_size = 0
        self.imgs = []

    def __call__(self, images):
        # if doesnt use sample_buffer, simply return the image
        if self.history_size == 0:
            return images
        output_imgs = []

        # If use sample buffer
        for image in images:
            # create a first dimension so that we will use torch.cat later
            image_mod = torch.unsqueeze(image,0)
            if self.current_size < self.history_size:
                self.imgs.append(image_mod)
                self.current_size += 1
                output_imgs.append(image_mod)
            else:
            # 50% use the image directly, 50% use the image from image buffer 
                if np.random.uniform(0,1)>0.5:
                    idx = np.random.randint(0, self.history_size - 1)
                    temp = self.imgs[idx].clone()
                    self.imgs[idx] = image_mod
                    output_imgs.append(temp)
                else:
                    output_imgs.append(image_mod)
        output_imgs = torch.cat(output_imgs, 0)
        return output_imgs

class linearLR():
    def __init__(self, total_epoch=200, decay_epoch=100):
        self.total_epoch = total_epoch
        self.decay_epoch = decay_epoch

    def get_lr(self, epoch):
        if epoch<self.decay_epoch:
            lr_coefficient = 1
        else:
            lr_coefficient = (epoch-self.decay_epoch)/float(self.total_epoch-self.decay_epoch)
        return lr_coefficient



def net_initialization(layer):  # define the initialization function
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 :  # if it is convolutional layer
        layer.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:  # if it is linear layer
        layer.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1: 
        layer.weight.data.normal_(1.0, 0.02)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0)

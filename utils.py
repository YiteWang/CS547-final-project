import os
import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def require_grad(nets, requires_grad=False):
    for net in nets:
        for params in net.parameters():
            param.requires_grad = requires_grad

class Sample_from_history(object):
    def __init__(self, history_size=50):
        self.history_size = history_size
        self.current_size = 0
        self.imgs = []

    def __call__(self, images):
        if self.history_size == 0:
            return images
        output_imgs = []
        for image in images:
            if self.current_size < self.history_size:
                self.imgs.append(image)
                self.current_size += 1
                output_imgs.append(image)
            else:
                if np.random.uniform(0,1)>0.5:
                    idx = np.random.randint(0, self.history_size - 1)
                    temp = self.imgs[idx].clone()
                    self.imgs[idx] = image
                    output_imgs.append(temp)
                else:
                    output_imgs.append(image)
        output_imgs = torch.cat(output_imgs, 0)
        return output_imgs

class linearLR(object):
     """docstring for linearLR"""
    def __init__(self, total_epoch=200, decay_epoch=100):
        self.total_epoch = total_epoch
        self.decay_epoch = decay_epoch

    def get_lr(self, epoch):
        lr = 1-max(0, epoch-self.decay_epoch)/float(self.total_epoch-self.decay_epoch)
        return lr

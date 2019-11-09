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
import torch
import torch.nn as nn
from torch.nn import init

def PatchDiscriminator(nn.Module):
    def __init__(self, input_channel, num_f=64, n_patch_layer=3, norm_layer=nn.BatchNorm2d, dropout_on=False, bias_on=False):
        super(PatchDiscriminator, self).__init__()
        kernel_size = 4
        padding_size = 1
        dis_arch = [nn.Conv2d(input_channel, num_f, kernel_size, stride=2, padding=padding_size),
                    nn.LeakyReLU(0.2, True)]

        num_f_multi_prev = 1
        num_f_multi_current = 1
        for i in range(1, n_patch_layer):
            num_f_multi_prev = num_f_multi_current
            num_f_multi_current = min(2**i, 8)
            dis_arch +=[nn.Conv2d(num_f_multi_prev*num_f, num_f_multi_current*num_f, kernel_size=kernel_size, stride=2, padding=padding_size, bias=bias_on),
                        norm_layer(num_f_multi_current*num_f),
                        nn.LeakyReLU(0.2, True)]
        
        num_f_multi_prev = num_f_multi_current
        num_f_multi_current = min(2**n_patch_layer, 8)
        dis_arch += [nn.Conv2d(num_f_multi_prev*num_f, num_f_multi_current*num_f, kernel_size = kernel_size, stride=1, padding=padding_size, bias=bias_on),
                     norm_layer(num_f_multi_current*num_f),
                     nn.LeakyReLU(0.2, True)]

        dis_arch +=  [nn.Conv2d(num_f_multi_current*num_f, 1, kernel_size=kernel_size, stride=1, padding=padding_size)]
        self.dis_arch = nn.Sequential(*dis_arch)

    def forward(self, input):
        return self.dis_arch(input)

def create_Discriminator(input_channel, num_f, norm='batch', n_patch_layer=3, dropout_on=False, bias_on=False, device='cpu')
    if norm_type =='batch':
        norm_layer = nn.BatchNorm2d

    Discriminator = PatchDiscriminator(input_channel, num_f, n_patch_layer, norm_layer, dropout_on, bias_on=False).to(device)
    return Discriminator

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

class Resnetblock(nn.Module):
    # Define a Resnetblock
    def __init__(self, dim, pad_method = 'reflect', norm_layer, dropout_on = False, baise_on = False):
        super(Resnetblock, self).__init__()
        resblock_arch = []
        if pad_method == 'zero':
            resblock_arch += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        elif pad_method == 'reflect':
            resblock_arch += [nn.ReflectionPad2d(1)]
            resblock_arch += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        elif pad_method == 'replicate':
            resblock_arch += [nn.ReplicationPad2d(1)]
            resblock_arch += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        if dropout_on:
            resblock_arch += [nn.Dropout(0.5)]

        if pad_method == 'zero':
            resblock_arch += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim)]
        elif pad_method == 'reflect':
            resblock_arch += [nn.ReflectionPad2d(1)]
            resblock_arch += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        elif pad_method == 'replicate':
            resblock_arch += [nn.ReplicationPad2d(1)]
            resblock_arch += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]

        self.resblock = nn.Sequential(*resblock_arch)
    
    def forward(self,x):
        return x + self.resblock(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_channel, output_channel, num_f=64, norm_layer=nn.BatchNorm2d, dropout_on=False, bias_on=False, num_block=6, pad_method='reflect'):
        super(ResnetGenerator, self).__init__
        Generator = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_channel, num_f, kernel_size=7, padding=0, bias=bias_on),
                     norm_layer(num_f),
                     nn.ReLU(True)]
        # Downsample twice
        Generator += [nn.Conv2d(num_f, num_f*2, 3, 2, 1, bias_on),
                      norm_layer(num_f*2),
                      nn.ReLU(True)]

        Generator += [nn.Conv2d(num_f*2, num_f*4, 3, 2, 1, bias_on),
                      norm_layer(num_f*4),
                      nn.ReLU(True)]       

        for i in range(num_block):
            Generator += [Resnetblock(num_f*4, norm_layer, dropout_on, bias_on)]

        Generator += [nn.ConvTranspose2d(num_f*4, num_f*2, 3, 2, 1, 1, bias_on),
                      norm_layer(num_f*2),
                      nn.ReLU(True)]

        Generator += [nn.ConvTranspose2d(num_f*2, num_f, 3, 2, 1, 1, bias_on),
                      norm_layer(num_f),
                      nn.ReLU(True)]  

        Generator += [nn.ReflectionPad2d(3)]
        Generator += [nn.Conv2d(num_f, output_channel, kernel_size=7, padding=0)]
        Generator += [nn.Tanh()]
        self.Generator = nn.Sequential(*Generator)

    def forward(self, input):
        return self.Generator(input)

def create_Generator(input_channel, output_channel, num_f, NN_name, norm='batch', dropout_on=False, device='cpu')
    if norm_type =='batch':
        norm_layer = nn.BatchNorm2d

    if NN_name == 'resnet9':
        Generator = ResnetGenerator(input_channel, output_channel, num_f, norm_layer, dropout_on, num_block=9).to(device)
    elif NN_name == 'resnet6':
        Generator = ResnetGenerator(input_channel, output_channel, num_f, norm_layer, dropout_on, num_block=6).to(device)
    return Generator

def init_weights(net, init_gain=0.02):

    def init_function(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_function) 
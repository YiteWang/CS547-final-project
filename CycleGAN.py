from arch import *
import os
import itertools
import functools
import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class cycleGAN(object):
    """docstring for cycleGAN"""
    def __init__(self, args):
        # super(cycleGAN, self).__init__()

        # Take arguments
        self.args = args

        # Defining generators
        self.Gxy = create_Generator(input_channel=3, output_channel=3, num_f=args.num_f, NN_name=args.NN_name, norm='batch', dropout_on=False, device='cuda')
        self.Gyx = create_Generator(input_channel=3, output_channel=3, num_f=args.num_f, NN_name=args.NN_name, norm='batch', dropout_on=False, device='cuda')
        
        # Defining discriminators
        self.Dx = create_Discriminator(input_channel=3, num_f=args.num_f, norm='batch', n_patch_layer=args.n_patch_layer, dropout_on=False, bias_on=False, device='cuda')
        self.Dy = create_Discriminator(input_channel=3, num_f=args.num_f, norm='batch', n_patch_layer=args.n_patch_layer, dropout_on=False, bias_on=False, device='cuda')
        
        self.MSE_l = nn.MSELoss()
        self.L1_l = nn.L1Loss()

        self.g_opt = torch.optim.Adam(itertools.chain(self.Gxy.parameters(),self.Gyx.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(itertools.chain(self.Gxy.parameters(),self.Gyx.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_opt, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        if args.avail_checkpoint = True:
            try:
                temp = utils.load_checkpoint('%s/latest.state' % (args.checkpoint_dir))
                self.start_epoch = temp['epoch']
                self.Dx.load_state_dict(temp['Dx'])
                self.Dy.load_state_dict(temp['Dy'])
                self.Gxy.load_state_dict(temp['Gxy'])
                self.Gyx.load_state_dict(temp['Gyx'])
                self.d_opt.load_state_dict(temp['d_opt'])
                self.g_opt.load_state_dict(temp['g_opt'])
            except:
                print('No checkpoint found, start from normal')
                self.start_epoch = 0
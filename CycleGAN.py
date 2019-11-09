from arch import *
import os
import itertools
import functools
import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils

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
        
        if args.GAN_name == 'vanilla':
            self.GAN_losscriterion = nn.BCEWithLogitsLoss()
        else: # default set least square/LSGAN
            self.GAN_losscriterion = nn.MSELoss()
        self.cycle_losscriterion = nn.L1Loss()

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
        else:
            self.start_epoch = 0

    def start_train(self,args):

        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             # transforms.RandomVerticalFlip(),
             transforms.Resize((args.load_H,args.load_W)),
             transforms.RandomCrop((args.crop_H,args.crop_W)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = {}
        dataset_dirs['trainX'] = os.path.join(args.dataset_dirs, 'trainA')
        dataset_dirs['trainY'] = os.path.join(args.dataset_dirs, 'trainB')
        x_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs['trainX'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        y_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs['trainY'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        device = args.device

        # a_fake_sample = []?

        for epoch in range(self.args.start_epoch, args.epochs):

            LR = self.g_opt.parameter_groups[0]['lr']

            for batch_idx, (x_real, y_real) in enumerate(zip(x_loader, y_loader)):
                ### First update generator
                # step = epoch * min(len(a_loader), len(b_loader)) + i + 1

                utils.require_grad([self.Dx, self.Dy], False)
                self.g_opt.zero_grad()
                x_real = torch.Tensor(x_real[0]).to(device) # x_real[1] is the class, default:0
                y_real = torch.Tensor(y_real[0]).to(device)

                # Forward passes
                x_fake_y = self.Gxy(x_real) # Produce Y using X
                y_fake_x = self.Gyx(y_real) # Produce X using Y

                x_y_x = self.Gyx(x_fake_y) # Reconstruct X
                y_x_y = self.Gxy(y_fake_x) # Reconstruct Y

                # GAN loss of x
                dis_x_fake_y = self.Dy(x_fake_y)
                dis_y_fake_x = self.Dx(y_fake_x)
                label_true = torch.ones(x_fake_y.size())
                label_false = torch.zeros(x_fake_y.size())
                x_GAN_loss = self.GAN_losscriterion(dis_y_fake_x, label_true) # if Dx can distinguish fake images
                y_GAN_loss = self.GAN_losscriterion(dis_x_fake_y, label_true) # if Dy can distinguish fake images

                # Cycle loss
                x_cycle_loss = self.cycle_losscriterion(x_real, x_y_x) * args.Lamda
                y_cycle_loss = self.cycle_losscriterion(y_real, y_x_y) * args.Lamda

                generator_loss = x_GAN_loss + y_GAN_loss + x_cycle_loss + y_cycle_loss

                generator_loss.backward()
                self.g_opt.step()

                ## The update discriminator


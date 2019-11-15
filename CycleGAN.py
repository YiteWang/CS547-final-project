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
        self.Gxy = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', dropout_on=False, device='cuda')
        self.Gyx = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', dropout_on=False, device='cuda')
        
        # Defining discriminators
        self.Dx = create_Discriminator(input_channel=3, num_f=args.num_c_d, norm='instance', n_patch_layer=args.n_patch_layer, dropout_on=False, bias_on=True, device='cuda')
        self.Dy = create_Discriminator(input_channel=3, num_f=args.num_c_d, norm='instance', n_patch_layer=args.n_patch_layer, dropout_on=False, bias_on=True, device='cuda')
        
        if args.GAN_name == 'vanilla':
            self.GAN_losscriterion = nn.BCEWithLogitsLoss()
        else: # default set least square/LSGAN
            self.GAN_losscriterion = nn.MSELoss()
        self.cycle_losscriterion = nn.L1Loss()

        self.g_opt = torch.optim.Adam(itertools.chain(self.Gxy.parameters(),self.Gyx.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(itertools.chain(self.Dx.parameters(),self.Dy.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda=utils.linearLR(args.epochs, args.decay_epoch).get_lr)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_opt, lr_lambda=utils.linearLR(args.epochs, args.decay_epoch).get_lr)

        if args.load_checkpoint = True:
            try:
                temp = torch.load('%s/latest.state' % (args.checkpoint_dir))
                self.start_epoch = temp['epoch']
                self.Dx.load_state_dict(temp['Dx'])
                self.Dy.load_state_dict(temp['Dy'])
                self.Gxy.load_state_dict(temp['Gxy'])
                self.Gyx.load_state_dict(temp['Gyx'])
                self.d_opt.load_state_dict(temp['d_opt'])
                self.g_opt.load_state_dict(temp['g_opt'])
                print('Checkpoint found, loaded successfully!')
            except:
                print('No checkpoint found, start from normal!')
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

        # a class that contains history(default size = 50) of using X to fake Y
        x_fake_y_history = utils.Sample_from_history() 
        # a class that contains history(default size = 50) of using Y to fake X
        y_fake_x_history = utils.Sample_from_history()

        for epoch in range(self.start_epoch, args.epochs):

            LR = self.g_opt.parameter_groups[0]['lr']

            for batch_idx, (x_real, y_real) in enumerate(zip(x_loader, y_loader)):
                ### First update generator
                # step =  batch_idx + 1 + min(len(x_loader), len(y_loader)) * epoch

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
                label_true = torch.ones(dis_x_fake_y.size())
                label_fake = torch.zeros(dis_x_fake_y.size())
                x_GAN_loss = self.GAN_losscriterion(dis_y_fake_x, label_true) # if Dx can distinguish fake images
                y_GAN_loss = self.GAN_losscriterion(dis_x_fake_y, label_true) # if Dy can distinguish fake images

                # Cycle loss
                x_cycle_loss = self.cycle_losscriterion(x_real, x_y_x) * args.lamda
                y_cycle_loss = self.cycle_losscriterion(y_real, y_x_y) * args.lamda

                generator_loss = x_GAN_loss + y_GAN_loss + x_cycle_loss + y_cycle_loss

                generator_loss.backward()
                self.g_opt.step()

                ## The update discriminator

                utils.require_grad([self.Dx, self.Dy], True)
                self.d_opt.zero_grad()

                x_fake = torch.Tensor(y_fake_x_history(y_fake_x.cpu().data.numpy())).to(device)
                y_fake = torch.Tensor(x_fake_y_history(x_fake_y.cpu().data.numpy())).to(device)

                dis_x_real = self.Dx(x_real)
                dis_y_real = self.Dy(y_real)
                dis_x_fake = self.Dx(x_fake)
                dis_y_fake = self.Dy(y_fake)

                # Discriminator loss
                x_dis_loss = 0.5 * (self.GAN_losscriterion(dis_x_fake, label_fake) + self.GAN_losscriterion(dis_x_real, label_true))
                y_dis_loss = 0.5 * (self.GAN_losscriterion(dis_y_fake, label_fake) + self.GAN_losscriterion(dis_y_real, label_true))

                x_dis_loss.backward()
                y_dis_loss.backward()
                self.d_opt.step()
                print("End of Epoch %3d, Batch: %5d/%5d | Loss of Gen:%.2e | Loss of Dis:%.2e" % (epoch, batch_idx + 1, min(len(x_loader), len(y_loader)), generator_loss, x_dis_loss+y_dis_loss))

            # save temp state
            save_param_dict = {'epoch': epoch+1,
                               'Dx': self.Dx.state_dict(),
                               'Dy': self.Dy.state_dict(),
                               'Gxy': self.Gxy.state_dict(),
                               'Gyx': self.Gyx.state_dict(),
                               'd_opt': self.d_opt.state_dict(),
                               'g_opt': self.g_opt.state_dict()}
            torch.save(save_param_dict, '%s/latest.state' % (args.checkpoint_dir))
            
            # learning rate scheduler
            self.g_scheduler.step()
            self.d_scheduler.step()
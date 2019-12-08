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
import numpy as np
from torch.autograd import Variable
import test

class cycleGAN(object):
    """docstring for cycleGAN"""
    def __init__(self, args):
        # super(cycleGAN, self).__init__()

        # Take arguments
        self.args = args

        # Defining generators
        self.Gxy = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', device='cuda')
        self.Gyx = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', device='cuda')
        
        # Defining discriminators
        self.Dx = create_Discriminator(input_channel=3, num_f=args.num_c_d, norm='instance', n_patch_layer=args.n_patch_layer, bias_on=True, device='cuda')
        self.Dy = create_Discriminator(input_channel=3, num_f=args.num_c_d, norm='instance', n_patch_layer=args.n_patch_layer, bias_on=True, device='cuda')
        
        if args.GAN_name == 'vanilla':
            self.GAN_losscriterion = nn.BCEWithLogitsLoss()
        else: # default set least square/LSGAN
            self.GAN_losscriterion = nn.MSELoss()
        self.cycle_losscriterion = nn.L1Loss()

        if args.use_id_loss:
            self.id_loss = nn.L1Loss()

        self.g_opt = torch.optim.Adam(itertools.chain(self.Gxy.parameters(),self.Gyx.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(itertools.chain(self.Dx.parameters(),self.Dy.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_opt, lr_lambda=utils.linearLR(args.epochs, args.decay_epoch).get_lr)
        self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_opt, lr_lambda=utils.linearLR(args.epochs, args.decay_epoch).get_lr)

        if args.load_checkpoint == True:
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

        x_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(args.dataset_dir, 'trainA'), transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        y_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(args.dataset_dir, 'trainB'), transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
        device = args.device

        # a class that contains history(default size = 50) of using X to fake Y
        x_fake_y_history = utils.Sample_from_history() 
        # a class that contains history(default size = 50) of using Y to fake X
        y_fake_x_history = utils.Sample_from_history()

        # This records the GAN loss of generator Gxy: X --> Y and discriminator DY
        GAN_loss_record_Gxy = []
        # This records the GAN loss of generator Gyx: Y --> X and discriminator DX
        GAN_loss_record_Gyx = []
        # This records the Cycle loss
        Cycle_loss_record = []
        # This records the Identity loss
        Gen_loss_record = []
        Dis_loss_record = []

        if args.use_id_loss:
            Identity_loss_record = []

        for epoch in range(self.start_epoch, args.epochs):
            epoch_GAN_Gxy = 0
            epoch_GAN_Gyx = 0
            epoch_Cycle_loss = 0
            epoch_Gen_loss = 0
            epoch_Dis_loss = 0
            if args.use_id_loss:
                epoch_Identity_loss = 0
            for batch_idx, (x_real, y_real) in enumerate(zip(x_loader, y_loader)):
                # First update generator

                utils.require_grad([self.Dx, self.Dy], False)
                self.g_opt.zero_grad()
                x_real = torch.Tensor(x_real[0]).to(device) # x_real[1] is the class, default:0
                y_real = torch.Tensor(y_real[0]).to(device)

                # Forward passes
                x_fake_y = self.Gxy(x_real) # Produce fake Y using X
                y_fake_x = self.Gyx(y_real) # Produce fake X using Y

                x_y_x = self.Gyx(x_fake_y) # Reconstruct X
                y_x_y = self.Gxy(y_fake_x) # Reconstruct Y

                # GAN loss of x: according to section 3.1, section 4 training details part.
                dis_x_fake_y = self.Dy(x_fake_y)
                dis_y_fake_x = self.Dx(y_fake_x)
                label_true = torch.ones(dis_x_fake_y.size()).to(device)
                label_fake = torch.zeros(dis_x_fake_y.size()).to(device)
                x_GAN_loss = self.GAN_losscriterion(dis_y_fake_x, label_true) # if Dx can distinguish fake images
                y_GAN_loss = self.GAN_losscriterion(dis_x_fake_y, label_true) # if Dy can distinguish fake images

                # Identity loss: According to section 5.2: Photo generation from painting part

                if args.use_id_loss:
                    y_identity = self.Gxy(y_real)
                    x_identity = self.Gyx(x_real)
                    y_id_loss = self.id_loss(y_identity, y_real) * args.lambda_id_loss
                    x_id_loss = self.id_loss(x_identity, x_real) * args.lambda_id_loss

                # Cycle loss According to section 3.2 of original paper
                x_cycle_loss = self.cycle_losscriterion(x_real, x_y_x) * args.lamda
                y_cycle_loss = self.cycle_losscriterion(y_real, y_x_y) * args.lamda

                generator_loss = x_GAN_loss + y_GAN_loss + x_cycle_loss + y_cycle_loss

                if args.use_id_loss:
                    generator_loss += y_id_loss + x_id_loss

                generator_loss.backward()
                self.g_opt.step()

                ## Then update discriminator

                utils.require_grad([self.Dx, self.Dy], True)
                self.d_opt.zero_grad()

                x_fake = Variable(y_fake_x_history(y_fake_x)).to(device)
                y_fake = Variable(x_fake_y_history(x_fake_y)).to(device)

                dis_x_real = self.Dx(x_real)
                dis_y_real = self.Dy(y_real)
                dis_x_fake = self.Dx(x_fake)
                dis_y_fake = self.Dy(y_fake)

                # Discriminator loss
                # This tells how good is Gyx (Compare Gyx(Y) with X)
                x_dis_loss =  (self.GAN_losscriterion(dis_x_fake, label_fake) + self.GAN_losscriterion(dis_x_real, label_true))
                # This tells how good is Gxy (Compare Gxy(X) with Y)
                y_dis_loss =  (self.GAN_losscriterion(dis_y_fake, label_fake) + self.GAN_losscriterion(dis_y_real, label_true))

                x_dis_loss.backward()
                y_dis_loss.backward()
                
                self.d_opt.step()
                if (batch_idx+1)%100 == 0 or (batch_idx + 1) == min(len(x_loader), len(y_loader)):
                    print("End of Epoch %d, Batch: %d/%d , Loss of Gen:%.2e , Loss of Dis:%.2e" % (epoch, batch_idx + 1, min(len(x_loader), len(y_loader)), generator_loss, x_dis_loss+y_dis_loss))
                
                epoch_GAN_Gxy += y_dis_loss.item() 
                epoch_GAN_Gyx += x_dis_loss.item()
                epoch_Cycle_loss += (x_cycle_loss + y_cycle_loss).item()
                epoch_Gen_loss += generator_loss.item()
                epoch_Dis_loss += (x_dis_loss + y_dis_loss).item()
                if args.use_id_loss:
                    epoch_Identity_loss += (y_id_loss + x_id_loss).item()

            # Store losses after each epoch
            GAN_loss_record_Gxy.append(epoch_GAN_Gxy)
            GAN_loss_record_Gyx.append(epoch_GAN_Gyx)
            Cycle_loss_record.append(epoch_Cycle_loss)
            Gen_loss_record.append(epoch_Gen_loss)
            Dis_loss_record.append(epoch_Dis_loss)

            # Save losses after each epoch
            np.save('%s/%s_GAN_Gxy.npy' % (args.checkpoint_dir, args.data_name), GAN_loss_record_Gxy)
            np.save('%s/%s_GAN_Gyx.npy' % (args.checkpoint_dir, args.data_name), GAN_loss_record_Gyx)
            np.save('%s/%s_Cycle_loss.npy' % (args.checkpoint_dir, args.data_name), Cycle_loss_record)
            np.save('%s/%s_Gen_loss.npy' % (args.checkpoint_dir, args.data_name), Gen_loss_record)
            np.save('%s/%s_Dis_loss.npy' % (args.checkpoint_dir, args.data_name), Dis_loss_record)
            if args.use_id_loss:
                Identity_loss_record.append(epoch_Identity_loss)
                np.save('%s/%s_Identity_loss.npy' % (args.checkpoint_dir, args.data_name), Identity_loss_record)

            # save temp state after each epoch
            save_param_dict = {'epoch': epoch+1,
                               'Dx': self.Dx.state_dict(),
                               'Dy': self.Dy.state_dict(),
                               'Gxy': self.Gxy.state_dict(),
                               'Gyx': self.Gyx.state_dict(),
                               'd_opt': self.d_opt.state_dict(),
                               'g_opt': self.g_opt.state_dict()}
            torch.save(save_param_dict, '%s/latest.state' % (args.checkpoint_dir))
            
            '''
            Save all the parameters every 50 epochs
            '''
            if (epoch+1)%50 == 0:
                torch.save(save_param_dict, '%s/%s.state' % (args.checkpoint_dir, str(epoch+1)))
                if args.test_in_train:
                    test.start_test(args, epoch+1)
            

            # learning rate scheduler
            self.g_scheduler.step()
            self.d_scheduler.step()

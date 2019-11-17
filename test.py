import os
import arch
import CycleGAN
import os
import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import *
from arch import *


def start_test(args):
    print('Start to test.')
    transform = transforms.Compose(
        [transforms.Resize((args.crop_H,args.crop_W)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = {}
    dataset_dirs['testX'] = os.path.join(args.dataset_dir, 'testA')
    dataset_dirs['testY'] = os.path.join(args.dataset_dir, 'testB')
    x_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs['testX'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)
    y_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs['testY'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)

    Gxy = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', dropout_on=False, device='cuda')
    Gyx = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', dropout_on=False, device='cuda')
    
    try:
        save_model = torch.load('%s/latest.state' % (args.checkpoint_dir))
        Gyx.load_state_dict(save_model['Gyx'])
        Gxy.load_state_dict(save_model['Gxy'])
    except:
        print('Checkpoint not found.')

    x_real = torch.Tensor(iter(x_loader).next()[0]).to(args.device)
    y_real = torch.Tensor(iter(y_loader).next()[0]).to(args.device)

    Gxy.eval()
    Gyx.eval()

    with torch.no_grad():
        y_fake = Gxy(x_real)
        x_fake = Gyx(y_real)
        x_recon = Gyx(y_fake)
        y_recon = Gxy(x_fake)

    pic = (torch.cat([x_real, y_fake, x_recon, y_real, x_fake, y_recon], dim=0).data + 1) / 2.0

    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    torchvision.utils.save_image(pic, args.result_dir+'/sample.jpg', nrow=3)


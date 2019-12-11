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


def start_test(args, epoch,test_all=False):
    print('Start to test.')
    transform = transforms.Compose(
        [transforms.Resize((args.crop_H*2,args.crop_W*2)), # resize to 512*512 for higher resolution
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # if test all, then dont shuffle. 
    shuffle_type = not test_all

    dataset_dirs_X = os.path.join(args.dataset_dir, 'testA')
    dataset_dirs_Y = os.path.join(args.dataset_dir, 'testB')
    x_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs_X, transform=transform), 
                                                        batch_size=args.test_batch_size, shuffle=True, num_workers=4)
    y_loader = torch.utils.data.DataLoader(datasets.ImageFolder(dataset_dirs_Y, transform=transform), 
                                                        batch_size=args.test_batch_size, shuffle=shuffle_type, num_workers=4)

    Gxy = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', device='cuda')
    Gyx = create_Generator(input_channel=3, output_channel=3, num_f=args.num_c_g, NN_name=args.gen_net, norm='instance', device='cuda')


    try:
        save_model = torch.load('%s/latest.state' % (args.checkpoint_dir))
        Gyx.load_state_dict(save_model['Gyx'])
        Gxy.load_state_dict(save_model['Gxy'])
    except:
        print('Checkpoint not found.')


    Gxy.eval()
    Gyx.eval()
    if test_all:
        for batch_idx, (x_real, y_real) in enumerate(zip(x_loader, y_loader)):
            x_real = torch.Tensor(x_real[0]).to(args.device)
            y_real = torch.Tensor(y_real[0]).to(args.device)
            with torch.no_grad():
                y_fake = Gxy(x_real)
                x_fake = Gyx(y_real)
                x_recon = Gyx(y_fake)
                y_recon = Gxy(x_fake)

            test_imge_output = (torch.cat([x_real, y_fake, x_recon, y_real, x_fake, y_recon], dim=0).data + 1) / 2.0

            if not os.path.isdir(args.result_dir):
                os.makedirs(args.result_dir)

            torchvision.utils.save_image(test_imge_output, args.result_dir+'/'+str(epoch)+'_batch_'+str(batch_idx)+'.jpg', nrow=args.test_batch_size)
    
    else:
        x_real = torch.Tensor(iter(x_loader).next()[0]).to(args.device)
        y_real = torch.Tensor(iter(y_loader).next()[0]).to(args.device)
        with torch.no_grad():
                y_fake = Gxy(x_real)
                x_fake = Gyx(y_real)
                x_recon = Gyx(y_fake)
                y_recon = Gxy(x_fake)
        test_imge_output = (torch.cat([x_real, y_fake, x_recon, y_real, x_fake, y_recon], dim=0).data + 1) / 2.0
        if not os.path.isdir(args.result_dir):
            os.makedirs(args.result_dir)

        torchvision.utils.save_image(test_imge_output, args.result_dir+'/'+str(epoch)+'.jpg', nrow=args.test_batch_size)
    






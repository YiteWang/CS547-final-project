import os
from argparse import ArgumentParser
from utils import *
import arch
import CycleGAN
import torch
import test

def retrieve_args():
    parser = ArgumentParser(description='CS547 Final Project: cycleGAN')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002, help='learning rate')
    parser.add_argument('--use_CPU', dest='use_GPU', action='store_false', help='change default device from GPU to CPU')
    parser.set_defaults(use_GPU=True)
    parser.add_argument('--use_id_loss', type=bool, default=False, help='if add identity loss')
    parser.add_argument('--lambda_id_loss', type=float, default=5, help='lambda used for identity loss')
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--load_H', type=int, default=286)
    parser.add_argument('--load_W', type=int, default=286)
    parser.add_argument('--crop_H', type=int, default=256)
    parser.add_argument('--crop_W', type=int, default=256)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--data_name', type=str, default='apple2orange', help='name of the datasets')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--num_c_g', type=int, default=64, help='# of channels in generator')
    parser.add_argument('--num_c_d', type=int, default=64, help='# of channels in discriminator')
    parser.add_argument('--gen_net', type=str, default='resnet9', help='type of generator')
    parser.add_argument('--n_patch_layer', type=int, default=3, help='number of patch layer')
    parser.add_argument('--GAN_name', type=str, default='lsgan', help='type of loss function of GANloss used')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--result_dir', type=str, default='./output_img/')
    parser.add_argument('--no-test_in_train', dest='test_in_train', action='store_false')
    parser.set_defaults(test_in_train=True)
    parser.add_argument('--no-GAN_loss', dest='use_GAN_loss', action='store_false')
    parser.set_defaults(use_GAN_loss=True)
    parser.add_argument('--no-cycle_loss', dest='use_cycle_loss', action='store_false')
    parser.set_defaults(use_cycle_loss=True)
    parser.add_argument('--no-forward_loss', dest='use_forward_loss', action='store_false')
    parser.set_defaults(use_forward_loss=True)
    parser.add_argument('--no-backward_loss', dest='use_backward_loss', action='store_false')
    parser.set_defaults(use_backward_loss=True)
    parser.add_argument('--test_batch_size', type=int, default=3)
    args = parser.parse_args()
    args.dataset_dir += args.data_name
    args.checkpoint_dir += args.data_name
    args.result_dir += args.data_name

    # create directory
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    return args


def main():
    args = retrieve_args()
    if args.use_GPU == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.device = device
    assert device != 'cpu', "Unable to find GPU! Stop running now."
    if args.training:
        print("Start to train")
        model = CycleGAN.cycleGAN(args)
        model.start_train(args)
    if args.testing:
        test.start_test(args, args.epochs, test_all=True)


if __name__ == '__main__':
    main()
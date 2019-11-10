import os
from argparse import ArgumentParser
from utils import *
import arch
import CycleGAN
import torch

def retrieve_args():
    parser = ArgumentParser(description='CS547 Final Project: cycleGAN')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--use_GPU', type=bool, default=True)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--load_H', type=int, default=286)
    parser.add_argument('--load_W', type=int, default=286)
    parser.add_argument('--crop_H', type=int, default=256)
    parser.add_argument('--crop_W', type=int, default=256)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--dataset_dir', type=str, default='./datasets/apple2orange')
    parser.add_argument('--load_checkpoint', type=bool, default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/apple2orange')
    parser.add_argument('--num_c_g', type=int, default=64, help='# of channels in generator')
    parser.add_argument('--num_c_d', type=int, default=64, help='# of channels in discriminator')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--GAN_name', type=str, default='vanilla')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
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
  # if args.testing:
  #     print("Testing")


if __name__ == '__main__':
    main()
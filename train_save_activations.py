from datasets import CIFAR10, MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import OtherNet, GroupNet, Net, GroupNetRGB, OtherNetRGB
from viz_experiment import get_activations
import argparse

# handle script arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save-step', type=int, default=10, metavar='N',
                    help='number of epochs to checkpoint the model')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--disable-pool', action='store_true', default=False,
                    help='enable maxpooling in the model')
parser.add_argument('--disable-dropout', action='store_true', default=False,
                    help='disable dropout in the model')
parser.add_argument('--kernel-size', type=int, default=3, metavar='K',
                    help='kernel size in the conv layers')
parser.add_argument('--comment', type=str, required=True,
                    help='run name')
model_parser = parser.add_mutually_exclusive_group(required=True)
model_parser.add_argument('--other', action='store_true', default=False)
model_parser.add_argument('--group', action='store_true', default=False)
model_parser.add_argument('--net', action='store_true', default=False)
model_parser.add_argument('--grouprgb', action='store_true', default=False)
model_parser.add_argument('--otherrgb', action='store_true', default=False)

dataset_parser = parser.add_mutually_exclusive_group(required=True)
dataset_parser.add_argument('--mnist', action='store_true', default=False)
dataset_parser.add_argument('--cifar10', action='store_true', default=False)
args = parser.parse_args()

num_epochs = args.epochs
save_step = args.save_step
kernel_size = args.kernel_size
maxpool = 1 if args.disable_pool else 2
dropout = not args.disable_dropout
if args.cifar10:
    dataset = CIFAR10(args.batch_size)
else:
    dataset = MNIST(args.batch_size)
comment = args.comment
if args.other:
    model_class = OtherNet
if args.group:
    model_class = GroupNet
if args.net:
    model_class = Net
if args.grouprgb:
    model_class = GroupNetRGB
model = model_class(dataset.batch_size, dataset.shape, kernel_size=kernel_size,
                 maxpool=maxpool, dropout=dropout)
if args.otherrgb:
    model = OtherNetRGB(dataset.batch_size, dataset.shape, kernel_size=kernel_size, m1=10, m2=30,
                        maxpool=maxpool, dropout=dropout)
print(args)

path = './' + model.__class__.__name__ + '_' + comment
optimizer = optim.SGD(model.parameters(), lr=args.lr,
        momentum=args.momentum)
for i in range(int(num_epochs/save_step)):
    model.train_with_loader(dataset.train_loader, dataset.test_loader, optimizer, num_epochs=save_step)
    model_path = path + '_epoch_' + str((i+1) * save_step) + '.torch'
    torch.save(model.state_dict(), model_path)
    img = dataset.get_random_examples(1)
    img_path = path + '_epoch_' + str((i+1) * save_step)
    get_activations(model, img, path = img_path)

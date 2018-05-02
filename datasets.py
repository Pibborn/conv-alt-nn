import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import CUDA_AVAILABLE

class MNIST():
    def __init__(self, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist-data', train=True, download=True,
                   transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                   ])),
                   batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist-data', train=False, download=True,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=batch_size, shuffle=True)
        self.batch_size = batch_size
        self.shape = (1, 28, 28)

    def get_random_examples(self, num_examples):
        imgs = torch.zeros([num_examples, 1, 28, 28])
        for i, (data, target) in enumerate(self.train_loader):
            imgs[i] = data[0]
            if i == num_examples - 1:
                break
        if CUDA_AVAILABLE:
            return Variable(imgs.cuda())
        else:
            return Variable(imgs)

class CIFAR10():
    def __init__(self, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./cifar-data', train=True,
            download=True, transform=transforms.ToTensor()), batch_size=batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./cifar-data', train=False,
            download=True, transform=transforms.ToTensor()), batch_size=batch_size,
            shuffle=True)
        self.batch_size = batch_size
        self.shape = (3, 32, 32)

    def get_random_examples(self, num_examples):
        imgs = torch.zeros([num_examples, 3, 32, 32])
        for i, (data, target) in enumerate(self.train_loader):
            imgs[i] = data[0]
            if i == num_examples - 1:
                break
        if CUDA_AVAILABLE:
            return Variable(imgs.cuda())
        else:
            return Variable(imgs)

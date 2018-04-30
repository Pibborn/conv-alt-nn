import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


mnist_train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist-data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=8, shuffle=True)

mnist_test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist-data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=8, shuffle=True)

cifar_train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar-data', train=True,
    download=True, transform=transforms.ToTensor()), batch_size=8, shuffle=True
)

cifar_test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar-data', train=False,
    download=True, transform=transforms.ToTensor()), batch_size=8, shuffle=True
)

def get_random_cifar_examples(num_examples):
    imgs = torch.zeros([num_examples, 3, 32, 32])
    for i, (data, target) in enumerate(cifar_train_loader):
        imgs[i] = data[0]
        if i == num_examples - 1:
            break
    return Variable(imgs)

def get_random_mnist_examples(num_examples):
    imgs = torch.zeros([num_examples, 1, 28, 28])
    for i, (data, target) in enumerate(mnist_train_loader):
        imgs[i] = data[0]
        if i == num_examples - 1:
            break
    return Variable(imgs)

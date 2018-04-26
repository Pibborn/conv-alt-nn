import torch
from torchvision import datasets, transforms

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

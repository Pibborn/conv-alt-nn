from datasets import CIFAR10, MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import OtherNet, GroupNet, Net

dataset = MNIST(4)
model = GroupNet(dataset.batch_size, dataset.shape)
path = './' + model.__class__.__name__ + '.torch'
optimizer = optim.SGD(model.parameters(), lr=0.01,
        momentum=0.5)
model.train_with_loader(dataset.train_loader, dataset.test_loader, optimizer, num_epochs=100)
torch.save(model.state_dict(), path)

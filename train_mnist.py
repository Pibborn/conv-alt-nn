from datasets import cifar_test_loader, cifar_train_loader, mnist_test_loader, mnist_train_loader
from datasets import get_random_mnist_examples
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import OtherNet, GroupNet

path = './groupnet.torch'
model = GroupNet()
optimizer = optim.SGD(model.parameters(), lr=0.01,
        momentum=0.5)
model.train_with_loader(mnist_train_loader, mnist_test_loader, optimizer, num_epochs=100)
torch.save(model.state_dict(), path)

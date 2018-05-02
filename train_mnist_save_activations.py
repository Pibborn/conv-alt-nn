from datasets import CIFAR10, MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import OtherNet, GroupNet, Net
from viz_experiment import get_activations

num_epochs = 100
save_step = 10

dataset = MNIST(8)
model = OtherNet(dataset.batch_size, dataset.shape)
path = './' + model.__class__.__name__
optimizer = optim.SGD(model.parameters(), lr=0.01,
        momentum=0.5)
for i in range(int(num_epochs/save_step)):
    model.train_with_loader(dataset.train_loader, dataset.test_loader, optimizer, num_epochs=save_step)
    model_path = path + '_epoch_' + str((i+1) * save_step) + '.torch'
    torch.save(model.state_dict(), model_path)
    img = dataset.get_random_examples(1)
    img_path = path + '_epoch_' + str((i+1) * save_step)
    get_activations(model, img, path = img_path)

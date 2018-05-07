from models import GroupNetRGB, Net, GroupNet, OtherNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datasets import MNIST

def get_activations(model, image, path=None, other=False):
    act_list = model.forward_return_activations(image)
    for i, layer_act in enumerate(act_list): # numero layer
        fig = plt.figure(figsize=(15, 15))
        split_depth = 3 if other else 1
        for j, act in enumerate(layer_act.split(split_depth, dim=0)): # attivazioni in un layer
            ax1 = fig.add_subplot(int(len(act_list[i])/5)+1, 6,j+1)
            activation = act.data.cpu().numpy()
            if other:
                activation = np.reshape(activation, activation.shape[::-1])
            else:
                activation = np.reshape(activation, activation.shape[:-1])
            ax1.imshow(activation, cmap='gray', shape=activation.shape)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        if path == None:
            plt.savefig(model.__class__.__name__ + '_activations_layer_' + (str(i)) + '.png')
        else:
            plt.savefig(path + '_activations_layer_' + (str(i)) + '.png')
        if other and i == 1:
            break

def create_probe_image(size, show=False):
    red = np.zeros((size, size))
    red[size//2, :] = 1
    green = np.zeros((size, size))
    green[:, size//2] = 1
    blue = np.eye(size)
    img = np.array([red, green, blue])
    #img = np.ravel(img, order='C')
    img = np.reshape(img, (-1, size, size))
    img = np.reshape(img, (size, size, -1))
    if show:
        plt.imshow(img)
        plt.show()
    return img

def connectivity_experiment():
    image = create_probe_image(32)
    image = torch.from_numpy(image).contiguous().float()
    #sys.exit(1)
    path = 'saved-models/groupnetrgb.torch'
    model = GroupNetRGB(1, (3, 32, 32), 3)
    model.test_connectivity(Variable(image.view(1, 3, 32, 32)))


if __name__ == '__main__':
    connectivity_experiment()
    sys.exit(1)
    torch.manual_seed(2)
    model = OtherNet(8, (1, 28, 28), kernel_size=3, maxpool=1)
    dataset = MNIST(8)
    img = dataset.get_random_examples(1)
    #model_path = 'models/' + model.__class__.__name__ + '.torch'
    model_path = 'models/OtherNet_kernel3_nopool_epoch_100.torch'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    get_activations(model, img)

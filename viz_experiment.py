from models import GroupNetRGB
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def get_activations(model, image):
    act_list = model.forward_return_activations(Variable(image.view(-1, 3, 32, 32)))
    for i, layer_act in enumerate(act_list): # numero layer
        fig = plt.figure()
        for j, act in enumerate(layer_act): # attivazioni in un layer
            ax1 = fig.add_subplot(int(len(act_list[i])/5), 5,j+1)
            ax1.imshow(act.data.numpy(), cmap='gray')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    plt.show()

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

if __name__ == '__main__':
    image = create_probe_image(32)
    image = torch.from_numpy(image).contiguous().float()
    #sys.exit(1)
    path = './groupnetrgb.torch'
    model = GroupNetRGB()
    model.forward_return_activations(Variable(image.view(1, 3, 32, 32)))

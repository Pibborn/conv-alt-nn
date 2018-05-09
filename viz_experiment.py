from models import GroupNetRGB, Net, GroupNet, OtherNet, OtherNetRGB
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datasets import MNIST, CIFAR10
import glob

def get_activations(model, image, path=None):
    act_list = model.forward_return_activations(image)
    for i, layer_act in enumerate(act_list): # numero layer
        fig = plt.figure(figsize=(15, 15))
        for j, act in enumerate(layer_act.split(1, dim=0)): # attivazioni in un layer
            ax1 = fig.add_subplot(int(len(act_list[i])/5)+1, 5, j+1)
            activation = act.data.cpu().numpy()
            activation = np.reshape(activation, activation.shape[::-1])
            activation = np.reshape(activation, activation.shape[:-1])
            ax1.imshow(activation, cmap='gray', shape=activation.shape)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        if path == None:
            plt.savefig(model.__class__.__name__ + '_activations_layer_' + (str(i)) + '.png')
        else:
            plt.savefig(path + '_activations_layer_' + (str(i)) + '.png')

def get_activations_rgb(model, image, path=None):
    act_list = model.forward_return_activations(image)
    for i, layer_act in enumerate(act_list): # numero layer
        fig = plt.figure(figsize=(15, 15))
        split_depth = 1
        img_shape = layer_act[0].shape
        img = np.array([]).reshape((*img_shape, 0)) #TODO
        for j, act in enumerate(layer_act.split(split_depth, dim=0)): # attivazioni in un layer
            activation = act.data.cpu().numpy()
            activation = np.reshape(activation, activation.shape[::-1])
            activation = np.reshape(activation, activation.shape[:-1])
            img = np.dstack([img, activation])
            if (j+1) % 3 == 0:
                ax1 = fig.add_subplot(int(len(act_list[i])/5)+1, 5, int(j/3)+1)
                try:
                    ax1.imshow(img, cmap='gray', shape=activation.shape)
                except TypeError:
                    break
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                img = np.array([]).reshape((*img_shape, 0))
        if path == None:
            plt.savefig(model.__class__.__name__ + '_activations_layer_' + (str(i)) + '.png')
        else:
            plt.savefig(path + '_activations_layer_' + (str(i)) + '.png')

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
    torch.manual_seed(1)
    dataset = CIFAR10(128)
    img = dataset.get_random_examples(1)
    #model_path = 'models/' + model.__class__.__name__ + '.torch'
    model_list = glob.glob('saved-models/*.torch')
    for i, model_name in enumerate(model_list):
        if 'noprune' in model_name:
            continue
        if i < 35:
            continue
        if 'OtherNet' in model_name and 'RGB' not in model_name:
            model = OtherNet(dataset.batch_size, dataset.shape, kernel_size=3)
        if 'GroupNet' in model_name and 'RGB' not in model_name:
            model = GroupNet(dataset.batch_size, dataset.shape, kernel_size=3)
        if 'Net' in model_name and 'Other' not in model_name and 'Group' not in model_name:
            model = Net(dataset.batch_size, dataset.shape, kernel_size=3)
        if 'GroupNetRGB' in model_name:
            model = GroupNetRGB(dataset.batch_size, dataset.shape, kernel_size=3)
        if 'OtherNetRGB' in model_name:
            model = OtherNetRGB(dataset.batch_size, dataset.shape, kernel_size=3, m1=10, m2=20)
        print(model_name)
        print(model.__class__.__name__)
        print(str(i+1) + '/' + str(len(model_list)))
        try:
            model.load_state_dict(torch.load(model_name, map_location='cpu'))
        except RuntimeError as ex:
            print(ex)
            continue
        path = model_name.strip('saved-models/').strip('.torch')
        get_activations(model, img, path = path + '_greyscale')
        get_activations_rgb(model, img, path = path + '_rgb')

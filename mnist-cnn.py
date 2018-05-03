from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
from scipy.ndimage import correlate
from scipy.signal import correlate2d
from utils import plot_weight_symmetry

# handle script arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
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
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--custom', dest='custom_model', action='store_true')
feature_parser.add_argument('--regular', dest='custom_model', action='store_false')
parser.add_argument('--model-path', help='Saved model path. If not set, a new model will be trained')
parser.set_defaults(custom_model=False)
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load and create data if needed
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist-data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist-data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class OtherNet(nn.Module):
    def __init__(self):
        super(OtherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=2)
        self.conv2_list = torch.nn.ModuleList()
        for i in range(5):
            self.conv2_list.append(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=2))
        self.conv2 = nn.Conv2d(150, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        a = torch.autograd.Variable(torch.ones(args.batch_size, 1, x.size()[2], x.size()[3]))
        first = True
        for i in range(5):
            for xi in x.split(1, dim=1):
                xi_a = F.relu(self.conv2_list[i](xi))
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(a)), 2))
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        activation_list.append(x[0][:][:])
        a = torch.autograd.Variable(torch.ones(args.batch_size, 1, x.size()[2], x.size()[3]))
        first = True
        for i in range(5):
            for xi in x.split(1, dim=1):
                xi_a = F.relu(self.conv2_list[i](xi))
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        activation_list.append(a[0][:][:])
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(a)), 2))
        activation_list.append(x[0][:][:])
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return activation_list



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        activation_list.append(x[0][:][:])
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        activation_list.append(x[0][:][:])
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return activation_list


def train(epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def plot_convmat(conv_mat_tensor, num_cols=5):
    # assumes shape (num_mat, 1, h, w)
    num_mat = conv_mat_tensor.shape[0]
    num_rows = math.ceil(num_mat / num_cols)
    fig = plt.figure(figsize=(num_cols,num_rows))
    for idx, conv_mat in enumerate(conv_mat_tensor):
        ax1 = fig.add_subplot(num_rows, num_cols, idx+1)
        ax1.imshow(conv_mat.reshape((conv_mat.shape[1], conv_mat.shape[2])), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def show_activations(conv_mat_tensor, image, num_cols=5):
    num_mat = conv_mat_tensor.shape[0]
    num_rows = math.ceil(num_mat / num_cols) + 1 # + 1 for original image
    fig = plt.figure(figsize=(num_cols,num_rows))
    for idx, conv_mat in enumerate(conv_mat_tensor):
        ax1 = fig.add_subplot(num_rows+1, num_cols, idx+num_cols+1)
        conv_mat = conv_mat.reshape((conv_mat.shape[1], conv_mat.shape[2]))
        activation = None
        #activation = correlate(image, conv_mat, output=activation, mode='constant')
        activation = correlate2d(image, conv_mat, mode='valid')
        ax1.imshow(activation, cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    ax1 = fig.add_subplot(num_rows+1, num_cols, math.ceil(num_cols//2)+1)
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])


    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def get_activations(model, image):
    act_list = model.forward_return_activations(image)
    for i, layer_act in enumerate(act_list): # numero layer
        fig = plt.figure()
        for j, act in enumerate(layer_act): # attivazioni in un layer
            ax1 = fig.add_subplot(int(len(act_list[i])/5), 5,j+1)
            ax1.imshow(act.data.numpy(), cmap='gray')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        plt.savefig(model.__class__.__name__ + '_activations_layer_' + (str(i)) + '.png')

def get_all_conv_mat(model):
    # for some reason the first module is the whole model
    conv_mat_dict = {}
    i = 1
    for module in model.modules():
        if 'conv' not in str(type(module)).lower():
            continue
        print(layer_to_str(module))
        temp_list = []
        weight_tensor = module.weight.data.numpy() # size: (in, out, w, h)
        for out_channel in weight_tensor:
            # in_channel size: (out, w, h)
            for conv_mat in out_channel:
                # conv_mat size: (w, h)
                temp_list.append(conv_mat)
        conv_mat_dict['conv'+str(i)] = temp_list
        i += 1
        plot_weight_symmetry(temp_list, 'sym-plots/conv'+str(i))
    return conv_mat_dict

def layer_to_str(module):
    return 'Shape: {}, name: {}'.format(module.weight.shape, str(type(module)))

if __name__ == '__main__':
    path = '/'
    if args.custom_model:
        print('Custom model.')
        model = OtherNet()
        path = './custom-model.torch'
    else:
        print('Regular model.')
        model = Net()
        path = './regular-model.torch'

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.model_path == None:
        # train a new model and save it
        loss_list = []
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            loss_list.append(test())
        print(loss_list)
        torch.save(model.state_dict(), path)
    else:
        model.load_state_dict(torch.load(args.model_path))
        first_layer_weights = model.conv1.weight.data.numpy()
        #plot_convmat(first_layer_weights)
        for idx, (data, target) in enumerate(train_loader):
            #img = data[0].numpy().reshape((data[0].shape[1], data[0].shape[2]))
            img = data[1]
            img = Variable(img.view(1, 1, 28, 28))
            break
        get_activations(model, img)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tensorboardX import SummaryWriter
import datetime

torch.manual_seed(10)
CUDA_AVAILABLE = torch.cuda.is_available()
time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# load and create data if needed
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist-data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=8, shuffle=True)

test_loader = torch.utils.data.DataLoader(
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

class BaseNet(ABC, nn.Module):

    @abstractmethod
    def __init__(self, batch_size, input_shape, kernel_size=3, maxpool=2, dropout=True):
        super(BaseNet, self).__init__()
        class_id = self.__class__.__name__
        self.writer = SummaryWriter(log_dir='runs/'+ class_id + '/' + time)
        self.global_step = 0
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.maxpool = maxpool
        self.dropout = dropout

    def train_with_loader(self, train_loader, test_loader, optimizer, num_epochs=10):
        self.train()
        if CUDA_AVAILABLE:
            self.cuda()
        train_list = []
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                if CUDA_AVAILABLE:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))
                self.writer.add_scalar('train_loss', loss, global_step=self.global_step)
                self.writer.add_scalar('test_loss', self.test_random_batch(test_loader),
                                       global_step=self.global_step)
                self.global_step += 1
            print('Training set:')
            train_loss, train_accuracy = self.test_with_loader(train_loader)
            print('Test set:')
            test_loss, test_accuracy = self.test_with_loader(test_loader)
            self.writer.add_scalar('test_accuracy', test_accuracy, global_step=self.global_step)
            self.writer.add_scalar('train_accuracy', train_accuracy, global_step=self.global_step)

    def test_random_batch(self, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        data, target = next(iter(test_loader))
        if CUDA_AVAILABLE:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = self(data)
        test_loss += F.nll_loss(output, target)
        self.train()
        return test_loss

    def get_linear_input_shape(self, last_conv_layer):
        if self.__class__.__name__ != 'OtherNet':
            f = last_conv_layer(Variable(torch.ones(1, *self.input_shape)))
            return int(np.prod(f.size()[1:]))
        else:
            raise NotImplementedError('Use the specific class method for OtherNet.')


    def test_with_loader(self, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if CUDA_AVAILABLE:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss, 100. * correct / len(test_loader.dataset)

class Net(BaseNet):
    def __init__(self, batch_size, input_shape, kernel_size=3, maxpool=2, dropout=True):
        super(Net, self).__init__(batch_size, input_shape)
        torch.manual_seed(10)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.maxpool1 = nn.MaxPool2d(maxpool)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(20, 50, kernel_size=kernel_size)
        self.maxpool2 = nn.MaxPool2d(maxpool)
        self.conv3_drop = nn.Dropout2d()
        self.features = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.conv3,
            self.maxpool2
        )
        self.fc_input_size = self.get_linear_input_shape(self.features)
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        if self.dropout:
            self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(self.maxpool1(self.conv1(x)))
        activation_list.append(x[0][:][:])
        x = F.relu(self.conv2(x))
        activation_list.append(x[0][:][:])
        x = self.conv3(x)
        if self.dropout:
            self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        activation_list.append(x[0][:][:])
        return activation_list

class OtherNet(BaseNet):
    def __init__(self, batch_size, input_shape, kernel_size=3, maxpool=2, dropout=True):
        super(OtherNet, self).__init__(batch_size, input_shape)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size, padding=2)
        self.maxpool1 = nn.MaxPool2d(maxpool)
        self.conv2_list = torch.nn.ModuleList()
        for i in range(5):
            self.conv2_list.append(nn.Conv2d(1, 3, kernel_size=kernel_size, padding=2))
        self.conv2 = nn.Conv2d(150, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(maxpool)
        self.conv2_drop = nn.Dropout2d()
        self.fc_input_size = self.othernet_get_linear_input_shape()
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        a = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(5):
            for xi in x.split(1, dim=1):
                xi_a = F.relu(self.conv2_list[i](xi))
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        x = self.conv2(a)
        if self.dropout:
            x = self.conv2_drop(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def othernet_get_linear_input_shape(self):
        x = torch.autograd.Variable(torch.ones(self.batch_size, *self.input_shape))
        x = self.maxpool1(self.conv1(x))
        a = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(5):
            for xi in x.split(1, dim=1):
                xi_a = self.conv2_list[i](xi)
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        x = self.conv2(a)
        if self.dropout:
            x = self.conv2_drop(x)
        x = F.relu(self.maxpool2(x))
        return int(np.prod(list(x.size())[1:]))

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(self.maxpool1(self.conv1(x)))
        activation_list.append(x[0][:][:])
        a = torch.autograd.Variable(torch.ones(1))
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
        x = self.conv2(a)
        if self.dropout:
            x = self.conv2_drop(x)
        x = F.relu(self.maxpool2(x))
        activation_list.append(x[0][:][:])
        return activation_list


class GroupNet(BaseNet):
    def __init__(self, batch_size, input_shape, kernel_size, maxpool=2, dropout=True):
        super(GroupNet, self).__init__(batch_size, input_shape, kernel_size)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size, padding=2)
        self.maxpool1 = nn.MaxPool2d(maxpool)
        self.conv2 = nn.Conv2d(10, 150, kernel_size=kernel_size, padding=2, groups=10)
        self.conv3 = nn.Conv2d(150, 20, kernel_size=kernel_size)
        self.maxpool2 = nn.MaxPool2d(maxpool)
        self.conv3_drop = nn.Dropout2d()
        self.features = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.conv3,
            self.maxpool2
        )
        self.fc_input_size = self.get_linear_input_shape(self.features)
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        if self.dropout:
            self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(self.maxpool1(self.conv1(x), 2))
        activation_list.append(x[0][:][:])
        x = F.relu(self.conv2(x))
        activation_list.append(x[0][:][:])
        x = self.conv3(x)
        if self.dropout:
            self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        activation_list.append(x[0][:][:])
        return activation_list


class GroupNetRGB(BaseNet):
    def __init__(self, batch_size, input_shape, kernel_size, maxpool=2, dropout=True):
        super(GroupNetRGB, self).__init__(batch_size, input_shape, kernel_size)
        self.conv1 = nn.Conv2d(3, 9, kernel_size=kernel_size, padding=0, groups=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(maxpool)
        self.conv2 = nn.Conv2d(9, 27, kernel_size=kernel_size, stride=1, padding=2, groups=3)
        self.maxpool2 = nn.MaxPool2d(maxpool)
        self.conv3 = nn.Conv2d(27, 10, kernel_size=kernel_size)
        self.conv3_drop = nn.Dropout2d()
        self.features = nn.Sequential(
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.conv3,
            self.maxpool2
        )
        self.fc_input_size = self.get_linear_input_shape(self.features)
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        if self.dropout:
            self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(self.maxpool1(self.conv1(x)))
        activation_list.append(x[0][:][:])
        x = F.relu(self.conv2(x))
        activation_list.append(x[0][:][:])
        x = self.conv3(x)
        if self.dropout:
            self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        activation_list.append(x[0][:][:])
        return activation_list

    def test_connectivity(self, x):
        convmat = torch.zeros(self.conv1.weight.size())
        convmat[:, :, 3//2, 3//2] = 1
        self.conv1.weight = torch.nn.Parameter(convmat)
        self.conv1.bias = torch.nn.Parameter(torch.zeros(self.conv1.bias.size()))
        activation_list = []
        x = F.relu(self.conv1(x))
        fig = plt.figure()
        for i, slice in enumerate(x[0][:][:][:]):
            slice = slice.data.numpy()
            slice = np.ravel(slice, order='C')
            slice = slice.reshape((30, 30), order='F')
            ax = fig.add_subplot(3, 3, i+1)
            ax.set_title('Slice {}'.format(i))
            ax.imshow(slice)
        plt.show(True)

class OtherNetRGB(BaseNet):
    def __init__(self, batch_size, input_shape, m1=3, m2=3, kernel_size=3, maxpool=2, dropout=True):
        super(OtherNetRGB, self).__init__(batch_size, input_shape)
        self.m1 = m1
        self.m2 = m2
        self.conv1_list = torch.nn.ModuleList()
        for i in range(m1):
            self.conv1_list.append(nn.Conv2d(3, 3, kernel_size=kernel_size, padding=int(kernel_size/2), groups=3))
        self.maxpool1 = nn.MaxPool2d(maxpool)
        self.conv2_list = torch.nn.ModuleList()
        for i in range(m2):
            self.conv2_list.append(nn.Conv2d(3, 3, kernel_size=kernel_size, padding=int(kernel_size/2), groups=3))
        self.conv3 = nn.Conv2d(600, 20, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(maxpool)
        self.conv3_drop = nn.Dropout2d()
        self.fc_input_size = self.othernetrgb_get_linear_input_shape()
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        a = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(self.m1):
            for xi in x.split(3, dim=1):
                xi_a = F.relu(self.maxpool1(self.conv1_list[i](xi)))
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        a2 = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(self.m2):
            for xi in a.split(3, dim=1):
                xi_a = F.relu(self.conv2_list[i](xi))
                if first:
                    a2 = xi_a
                    first = False
                else:
                    a2 = torch.cat((a2, xi_a), 1)
        x = a2
        x = self.conv3(x)
        if self.dropout:
            x = self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        a = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(self.m1):
            for xi in x.split(3, dim=1):
                xi_a = F.relu(self.maxpool1(self.conv1_list[i](xi)))
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        activation_list.append(a[0][:][:])
        first = True
        a2 = torch.autograd.Variable(torch.ones(1))
        for i in range(self.m2):
            for xi in a.split(3, dim=1):
                xi_a = F.relu(self.conv2_list[i](xi))
                if first:
                    a2 = xi_a
                    first = False
                else:
                    a2 = torch.cat((a2, xi_a), 1)
        activation_list.append(a2[0][:][:])
        x = a2
        x = self.conv3(x)
        if self.dropout:
            x = self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        activation_list.append(x[0][:][:])
        return activation_list

    def othernetrgb_get_linear_input_shape(self):
        x = torch.autograd.Variable(torch.ones(self.batch_size, *self.input_shape))
        a = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(self.m1):
            for xi in x.split(3, dim=1):
                xi_a = F.relu(self.conv1_list[i](xi))
                if first:
                    a = xi_a
                    first = False
                else:
                    a = torch.cat((a, xi_a), 1)
        a2 = torch.autograd.Variable(torch.ones(1))
        first = True
        for i in range(self.m2):
            for xi in a.split(3, dim=1):
                xi_a = F.relu(self.conv2_list[i](xi))
                if first:
                    a2 = xi_a
                    first = False
                else:
                    a2 = torch.cat((a2, xi_a), 1)
        x = a2
        x = self.conv3(x)
        if self.dropout:
            x = self.conv3_drop(x)
        x = F.relu(self.maxpool2(x))
        return int(np.prod(list(x.size())[1:]))


def get_activations(model, image):
    act_list = model.forward_return_activations(Variable(image.view(-1, 1, 28, 28)))
    for i, layer_act in enumerate(act_list): # numero layer
        fig = plt.figure()
        for j, act in enumerate(layer_act): # attivazioni in un layer
            ax1 = fig.add_subplot(int(len(act_list[i])/5), 5,j+1)
            ax1.imshow(act.data.numpy(), cmap='gray')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    plt.show()

def create_probe_image(size):
    red = np.zeros((size, size))
    red[size//2, :] = 1
    green = np.zeros((size, size))
    green[:, size//2] = 1
    blue = np.eye(size)
    img = np.array([red, green, blue])
    #img = np.ravel(img, order='C')
    img = np.reshape(img, (-1, size, size))
    img = np.reshape(img.T, (size, size, -1))
    plt.imshow(img)
    plt.show()
    return img

if __name__ == '__main__':
    model = GroupNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6)
    model.train_with_loader(train_loader, test_loader, optimizer, num_epochs=1)
    model.test_with_loader(test_loader)
    #path = './groupnetrgb.torch'
    #model = GroupNetRGB()
    #optimizer = optim.SGD(model.parameters(), lr=0.01,
    #    momentum=0.5)
#
    ##model.load_state_dict(torch.load(path))
#
    ##for epoch in range(1, 2):
    ##    train(model, epoch)
    ##torch.save(model.state_dict(), path)
#
    #for data, target in cifar_train_loader:
    #    img = data[0]
    #    break
    #model.forward_return_activations(Variable(image.view(1, 3, 32, 32)))
    ##get_activations(model, img)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(10)

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(10)

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
        a = torch.autograd.Variable(torch.ones(8, 1, x.size()[2], x.size()[3]))
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


class GroupNet(nn.Module):
    def __init__(self):
        super(GroupNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(10, 150, kernel_size=3, stride=1, padding=2, groups=10)
        self.conv3 = nn.Conv2d(150, 20, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        activation_list.append(x[0][:][:])
        x = F.relu(self.conv2(x))
        activation_list.append(x[0][:][:])
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        activation_list.append(x[0][:][:])
        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return activation_list

class GroupNetRGB(nn.Module):
    def __init__(self):
        super(GroupNetRGB, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, padding=0, groups=3, stride=1)
        #for convmat in self.conv1.weight:
        convmat = torch.zeros(self.conv1.weight.size())
        convmat[:, :, 3//2, 3//2] = 1
        self.conv1.weight = torch.nn.Parameter(convmat)
        self.conv1.bias = torch.nn.Parameter(torch.zeros(self.conv1.bias.size()))
        self.conv2 = nn.Conv2d(9, 27, kernel_size=3, stride=1, padding=2, groups=3)
        self.conv3 = nn.Conv2d(27, 10, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(490, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 490)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def forward_return_activations(self, x):
        activation_list = []
        x = F.relu(self.conv1(x))
        print(x.size())
        fig = plt.figure()
        for i, slice in enumerate(x[0][:][:][:]):
            slice = slice.data.numpy()
            print(slice)
            slice = np.ravel(slice, order='C')
            slice = slice.reshape((30, 30), order='F')
            ax = fig.add_subplot(3, 3, i+1)
            ax.set_title('Slice {}'.format(i))
            ax.imshow(slice)
        plt.show()
        activation_list.append(x[0][:][:])
        x = F.relu(self.conv2(x))
        activation_list.append(x[0][:][:])
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        activation_list.append(x[0][:][:])
        x = x.view(-1, 490)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return activation_list


def train(model, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(cifar_train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model):
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
    img = np.stack([red, green, blue])
    #img = np.ravel(img, order='C')
    #img = img.reshape((size, size, 3), order='F')
    return img

if __name__ == '__main__':
    image = create_probe_image(32)
    image = torch.from_numpy(image).contiguous().float()
    #sys.exit(1)
    path = './groupnetrgb.torch'
    model = GroupNetRGB()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
        momentum=0.5)

    #model.load_state_dict(torch.load(path))

    #for epoch in range(1, 2):
    #    train(model, epoch)
    #torch.save(model.state_dict(), path)

    for data, target in cifar_train_loader:
        img = data[0]
        break
    model.forward_return_activations(Variable(image.view(1, 3, 32, 32)))
    #get_activations(model, img)

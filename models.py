import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable

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
        torch.manual_seed(10)

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
        torch.manual_seed(10)

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

def train(model, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
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

if __name__ == '__main__':

    model = GroupNet()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    optimizer = optim.SGD(model.parameters(), lr=0.01,
        momentum=0.5)

    for epoch in range(1, 10):
        train(model, epoch)
        loss_list.append(test(model))
    print(loss_list)
    torch.save(model.state_dict(), path)

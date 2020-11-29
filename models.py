import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


class Actor(nn.Module):
    def __init__(self, height=546, width=735, channels=1, action_dim=4, control_dim=4): # action dim: x, y, right_click, left_click
        super(Actor, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(self.channels, 8, kernel_size=9, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 12, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(12)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width, kernel_size=9), kernel_size=7)), kernel_size=3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height, kernel_size=9), kernel_size=7)), kernel_size=3)

        self.fc1 = nn.Linear(convh * convw * 12, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fcc1 = nn.Linear(control_dim, 128)
        self.fcc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(512 + 256, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, state, controls_state):
        x = F.leaky_relu(self.bn1(self.conv1(state)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        y = F.leaky_relu(self.fcc1(controls_state))
        y = F.leaky_relu(self.fcc2(y))
        z = torch.cat((x, y), dim=1)
        z = F.leaky_relu(self.fc3(z))
        z = self.fc4(z)
        return z


class Critic(nn.Module):
    def __init__(self, height=546, width=735, channels=1, action_dim=4, control_dim=4):
        super(Critic, self).__init__()

        self.width = width
        self.height = height
        self.channels = channels
        self.action_dim = action_dim

        self.convs1 = nn.Conv2d(self.channels, 8, kernel_size=9, stride=2)
        self.bns1 = nn.BatchNorm2d(8)
        self.convs2 = nn.Conv2d(8, 16, kernel_size=7, stride=2)
        self.bns2 = nn.BatchNorm2d(16)
        self.convs3 = nn.Conv2d(16, 24, kernel_size=5, stride=2)
        self.bns3 = nn.BatchNorm2d(24)
        self.convs4 = nn.Conv2d(24, 12, kernel_size=3, stride=2)
        self.bns4 = nn.BatchNorm2d(12)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width, kernel_size=9), kernel_size=7)), kernel_size=3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height, kernel_size=9), kernel_size=7)), kernel_size=3)

        self.fcs4 = nn.Linear(convh * convw * 12, 2048)
        self.fcs5 = nn.Linear(2048, 512)

        self.fca1 = nn.Linear(self.action_dim, 128)
        self.fca2 = nn.Linear(128, 256)

        self.fc1 = nn.Linear(control_dim, 128)
        self.fc2 = nn.Linear(128, 256)

        self.ffc1 = nn.Linear(1024, 128)
        self.ffc2 = nn.Linear(128, 1)

    def forward(self, state, controls_state, action):  # Compute an approximate Q(s, a) value function
        x = F.leaky_relu(self.bns1(self.convs1(state)))
        x = F.leaky_relu(self.bns2(self.convs2(x)))
        x = F.leaky_relu(self.bns3(self.convs3(x)))
        x = F.leaky_relu(self.bns4(self.convs4(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fcs4(x))
        x = F.leaky_relu(self.fcs5(x))
        y = F.leaky_relu(self.fc1(controls_state))
        y = F.leaky_relu(self.fc2(y))
        z = F.leaky_relu(self.fca1(action))
        z = F.leaky_relu(self.fca2(z))
        x = torch.cat((x, y, z), dim=1)
        x = F.leaky_relu(self.ffc1(x))
        x = self.ffc2(x)
        return x



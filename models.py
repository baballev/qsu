import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.
counting = 0


class Actor(nn.Module):
    def __init__(self, height=137, width=184, channels=1, action_dim=4, control_dim=4): # action dim: x, y, right_click, left_click
        super(Actor, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(self.channels, 16, kernel_size=7, stride=2)
        #self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(16)
        #self.conv3 = nn.Conv2d(8, 12, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(16)
        #self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=2)
        #self.bn4 = nn.BatchNorm2d(8)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(width, kernel_size=7), kernel_size=5)
        convh = conv2d_size_out(conv2d_size_out(height, kernel_size=7), kernel_size=5)

        self.fc1 = nn.Linear(convh * convw * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fcc1 = nn.Linear(control_dim, 12)
        self.fc3 = nn.Linear(512 + 12, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, action_dim)

    def forward(self, state, controls_state):
        global counting
        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        #x = F.leaky_relu(self.conv3(x))

        if counting % 3000 == 0:
            transforms.ToPILImage()(state[0]).save('truc_state.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 0, :, :]).save('truc_0.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 1, :, :]).save('truc_1.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 2, :, :]).save('truc_2.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 3, :, :]).save('truc_3.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 4, :, :]).save('truc_4.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 5, :, :]).save('truc_5.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 6, :, :]).save('truc_6.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 7, :, :]).save('truc_7.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 8, :, :]).save('truc_8.' + str(counting) + '.png')

        #x = F.leaky_relu(self.conv4(x))
        #transforms.ToPILImage()(state[0]).save(str(counting) + '.png')
        #transforms.ToPILImage()(x[0, 0, :, :]).save('truc_0_' + str(counting) + '.png')
        counting += 1
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        y = F.leaky_relu(self.fcc1(controls_state))
        y = torch.cat((x, y), dim=1)
        y = F.leaky_relu(self.fc3(y))
        y = F.leaky_relu(self.fc4(y))
        y = F.leaky_relu(self.fc5(y))
        y = torch.tanh(self.fc6(y))
        return 0.5 * y + 0.5  # [-1, 1] --> [0, 1]


class Critic(nn.Module):
    def __init__(self, height=137, width=184, channels=1, action_dim=4, control_dim=4):
        super(Critic, self).__init__()

        self.width = width
        self.height = height
        self.channels = channels
        self.action_dim = action_dim

        self.convs1 = nn.Conv2d(self.channels, 16, kernel_size=7, stride=2)
        #self.bns1 = nn.BatchNorm2d(8)
        self.convs2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        #self.bns2 = nn.BatchNorm2d(16)
        #self.convs3 = nn.Conv2d(8, 12, kernel_size=5, stride=2)
        #self.bns3 = nn.BatchNorm2d(16)
        #self.convs4 = nn.Conv2d(12, 16, kernel_size=3, stride=2)
        #self.bns4 = nn.BatchNorm2d(8)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(width, kernel_size=7), kernel_size=5)
        convh = conv2d_size_out(conv2d_size_out(height, kernel_size=7), kernel_size=5)

        self.fcs4 = nn.Linear(convh * convw * 32, 512)
        self.fcs5 = nn.Linear(512, 256)

        self.fca1 = nn.Linear(self.action_dim, 32)

        self.fc1 = nn.Linear(control_dim, 12)

        self.ffc1 = nn.Linear(256+32+12, 1)

    def forward(self, state, controls_state, action):  # Compute an approximate Q(s, a) value function
        x = F.leaky_relu(self.convs1(state))
        x = F.leaky_relu(self.convs2(x))
        #x = F.leaky_relu(self.convs3(x))
        #x = F.leaky_relu(self.convs4(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fcs4(x))
        x = F.leaky_relu(self.fcs5(x))
        y = F.leaky_relu(self.fc1(controls_state))
        z = F.leaky_relu(self.fca1(action))
        x = torch.cat((x, y, z), dim=1)
        x = self.ffc1(x)
        return x


class QNetwork(nn.Module):

    def __init__(self, height=150, width=220, channels=4, action_dim=7400, control_dim=4):
        super(QNetwork, self).__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)

        self.fc3 = nn.Linear(convh * convw * 64, 1024)
        self.fcc3 = nn.Linear(control_dim, 64)
        self.fc4 = nn.Linear(1024+64, action_dim)

    def forward(self, state, control_state):
        global counting
        x = F.leaky_relu(self.conv1(state))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        '''
        if counting % 20 == 15:
            transforms.ToPILImage()(state[0]).save('truc_state.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 0, :, :]).save('truc_0.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 1, :, :]).save('truc_1.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 2, :, :]).save('truc_2.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 3, :, :]).save('truc_3.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 4, :, :]).save('truc_4.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 5, :, :]).save('truc_5.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 6, :, :]).save('truc_6.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 7, :, :]).save('truc_7.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 8, :, :]).save('truc_8.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 9, :, :]).save('truc_9.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 10, :, :]).save('truc_10.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 11, :, :]).save('truc_11.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 12, :, :]).save('truc_12.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 13, :, :]).save('truc_13.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 14, :, :]).save('truc_14.' + str(counting) + '.png')
            transforms.ToPILImage()(x[0, 15, :, :]).save('truc_15.' + str(counting) + '.png')
        counting += 1
        '''
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc3(x))
        y = F.leaky_relu(self.fcc3(control_state))
        y = torch.cat((x, y), dim=1)
        return self.fc4(y)

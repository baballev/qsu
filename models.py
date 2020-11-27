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

        self.conv1 = nn.Conv2d(self.channels, 8, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(8)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width))), kernel_size=3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height))), kernel_size=3)

        ## DEBUG
        print('Actor conv output size:')
        print('Conv output width: ' + str(convw))
        print('Conv output height: ' + str(convh))

        self.fc1 = nn.Linear(convh * convw * 8 + control_dim, 1024)
        self.fc2 = nn.Linear(1024, action_dim)

    def forward(self, state, controls_state):
        x = F.leaky_relu(self.bn1(self.conv1(state)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = torch.cat((x.view(x.size(0), -1), controls_state.view(controls_state.size(0), -1)), dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # TODO: TRY WITHOUT THE SIGMOID?
        return x * self.width


class Critic(nn.Module):

    def __init__(self, height=546, width=735, channels=1, action_dim=4, control_dim=4):
        super(Critic, self).__init__()

        self.width = width
        self.height = height
        self.channels = channels
        self.action_dim = action_dim

        self.convs1 = nn.Conv2d(self.channels, 8, kernel_size=5, stride=2)
        self.bns1 = nn.BatchNorm2d(8)
        self.convs2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.bns2 = nn.BatchNorm2d(16)
        self.convs3 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bns3 = nn.BatchNorm2d(16)
        self.convs4 = nn.Conv2d(16, 8, kernel_size=3, stride=2)
        self.bns4 = nn.BatchNorm2d(8)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(width))), kernel_size=3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(height))), kernel_size=3)

        ## DEBUG
        print('Critic conv output size for state: ')
        print('Conv output width: ' + str(convw))
        print('Conv output height: ' + str(convh))

        self.fcs4 = nn.Linear(convh * convw * 8 + control_dim, 1024)

        self.fca1 = nn.Linear(self.action_dim, 128)

        self.fc1 = nn.Linear(128 + 1024, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, controls_state, action):  # Compute an approximate Q(s, a) value function
        x = F.leaky_relu(self.bns1(self.convs1(state)))
        x = F.leaky_relu(self.bns2(self.convs2(x)))
        x = F.leaky_relu(self.bns3(self.convs3(x)))
        x = F.leaky_relu(self.bns4(self.convs4(x)))
        x = torch.cat((x.view(x.size(0), -1), controls_state.view(controls_state.size(0), -1)), dim=1)
        x = F.leaky_relu(self.fcs4(x))
        action = F.relu(self.fca1(action))

        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


## Not Used ATM


class CNN(nn.Module):  # ToDo: IMPORTANT: THIS Could be used for making a model of the environment? But not sure that it is necessary.
                       # ToDo: If we make a model of the environment using CNNs, I DO NOT NEED to use convolutions anymore in my actors and critics.
                       # ToDo: I have a simple CNN's output vector that represents the state of the game, gotten from the input of the CNN
                       # ToDo: which is the current screen of the game. If i want to do this i need to check how i backprop and differentiate through
                       # ToDo: this environment model.
    ''' Input: Image i.e the game screen
        Output: 1024 dim vector to be concatenated with other state info?
        Parameters: Image size and the number of dimensions in the output.
    '''

    def __init__(self, height=600, width=1024, outputs=1024): # ToDo number of outputs to determine
        super(CNN, self).__init__()
        # ToDo: Try to implement with ROI so that we get more relevant feature maps like in masking-RCNN?
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))


        self.fc = nn.Linear(convh * convw * 32, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(x.view(x.size(0), -1))
import torch
import torch.nn as nn
import torch.nn.functional as F


## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


class Actor(nn.Module):

    def __init__(self, height=600, width=1024, channels=3, action_dim=4): # action dim: x, y, right_click, left_click
        super(Actor, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(self.channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))

        ## DEBUG
        print('Actor conv output size:')
        print('Conv output width: ' + str(convw))
        print('Conv output height: ' + str(convh))


        self.fc = nn.Linear(convh * convw * 32, action_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.sigmoid(self.fc(x.view(x.size(0), -1))) * self.width


class Critic(nn.Module):

    def __init__(self, height=600, width=1024, channels=3, action_dim=4):
        super(Critic, self).__init__()

        self.width = width
        self.height = height
        self.channels = channels
        self.action_dim = action_dim

        self.convs1 = nn.Conv2d(self.channels, 16, kernel_size=5, stride=2)
        self.bns1 = nn.BatchNorm2d(16)
        self.convs2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bns2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(width))
        convh = conv2d_size_out(conv2d_size_out(height))

        ## DEBUG
        print('Critic conv output size for state: ')
        print('Conv output width: ' + str(convw))
        print('Conv output height: ' + str(convh))

        self.fcs3 = nn.Linear(convh * convw * 32, 128)

        self.fca1 = nn.Linear(self.action_dim, 128)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, action):  # Compute an approximate Q(s, a) value function
        state = F.relu(self.bns1(self.convs1(state)))
        state = F.relu(self.bns2(self.convs2(state)))
        state = F.relu(self.fcs3(state.view(state.size(0), -1)))

        action = F.relu(self.fca1(action))

        x = torch.cat((state, action), dim=1)
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
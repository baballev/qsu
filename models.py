
import torch.nn as nn
import torch.functional as F

class CNN(nn.Module):
    ''' Input: Image i.e the game screen
        Output: 1024 dim vector to be concatenated with other state info?
        Parameters: Image size and the number of dimensions in the output.
    '''

    def __init__(self, height=600, width=1024, outputs=1024): #ToDo number of outputs to determine
        super(DQN, self).__init__()
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
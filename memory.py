from collections import namedtuple
import random
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Based on pytorch DQN tutorial

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(torch.squeeze(args[0], 0), torch.squeeze(args[1], 0), args[2], torch.squeeze(args[3], 0))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s = torch.stack([a[0] for a in batch])
        a = torch.stack([a[1] for a in batch])
        r = torch.stack([a[2] for a in batch])
        s1 = torch.stack([a[3] for a in batch])
        return s, a, r, s1

    def __len__(self):
        return len(self.memory)

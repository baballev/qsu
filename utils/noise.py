import torch
import math
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000

## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.

t = 0


def action_noise(action_dim=4, mu=0, theta=0.15, sigma=0.2):
    global t
    decay_factor = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * t / EPS_DECAY)
    x = decay_factor * torch.randn(4)
    # X = torch.ones(action_dim)
    # dx = theta * (mu - X)
    # dx = dx + sigma * torch.randn(len(X))
    # X = X + dx
    # print('Noise')
    # print(x)
    t += 1
    return x
import torch


## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


def action_noise(action_dim=4, mu=0, theta=0.15, sigma=0.2):
    X = torch.ones(action_dim)
    dx = theta * (mu - X)
    dx = dx + sigma * torch.randn(len(X))
    X = X + dx
    return X
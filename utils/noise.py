import torch

def action_noise(action_dim=5, mu=0, theta=0.15, sigma=0.2):
    X = torch.ones(action_dim)
    dx = theta * (mu - X)
    dx = dx + sigma * torch.randn(len(X))
    X = X + dx
    return X
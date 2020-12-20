import torch
## BASED ON: https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass


class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma, min_val, max_val):
        self.mu = mu
        self.sigma = sigma
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self):
        return torch.clamp(self.sigma * torch.randn(3).to(device) + self.mu, self.min_val, self.max_val)


class OsuDiscreteNoise(NormalActionNoise):
    def __init__(self, mu, sigma, min_val, max_val):
        super(OsuDiscreteNoise, self).__init__(mu, sigma, min_val, max_val)

    def __call__(self):
        return torch.cat([torch.clamp(self.sigma * torch.randn(2).to(device) + self.mu, self.min_val, self.max_val), torch.rand(1).to(device)])

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=torch.tensor([0.12], device=device), x0=None, min_val=0.0, max_val=0.9999):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.min_val = min_val
        self.max_val = max_val
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(self.mu, device=device)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * torch.sqrt(self.dt) * torch.randn(size=self.mu.shape, device=device)
        self.x_prev = x
        return torch.clamp(x, self.min_val, self.max_val)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(self.mu, device=device)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

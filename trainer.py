import pyclick
import torch
import torch.nn.functional as F

import models
import utils.noise
from memory import ReplayMemory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.
def hard_copy(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):  # y = TAU * x  +  (1 - TAU) * y
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Trainer:

    def __init__(self, batch_size=5, lr=0.001, tau=0.0001, gamma=0.999, load_weights=None, width=1024, height=600): # ToDo: change width and height of state
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.actor = models.Actor().to(device)
        self.target_actor = models.Actor().to(device)
        self.critic = models.Critic().to(device)
        self.target_critic = models.Critic().to(device)
        if load_weights is not None:
            self.load_models(load_weights)
        else:
            hard_copy(self.target_actor, self.actor)
            hard_copy(self.target_critic, self.critic)

        self.noise = utils.noise.OrnsteinUhlenbeckActionNoise(mu=torch.tensor([512.0, 300.0, 512.0, 512.0]), sigma=200.0, theta=150, x0=torch.tensor([512.0, 300, 512.0, 512.0]))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.lr)

        self.memory = ReplayMemory(1500)  # The larger the better because then the transitions have more chances to be uncorrelated

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        self.ocr = utils.OCR.init_OCR()
        self.hc = pyclick.HumanClicker()

    def select_exploration_action(self, state):  # Check if the values are ok
        action = self.actor(state).detach()
        new_action = action + self.noise().to(device)  # ToDo: Check the noise progression with print or plot
        return new_action  # ToDO: Code exploitation policy action

    def save_model(self, file_name):
        torch.save(self.target_actor.state_dict(), './weights/actor' + file_name)
        print('Model saved to : ' + './weights/actor' + file_name)
        torch.save(self.target_critic.state_dict(), './weights/critic' + file_name)
        print('Model saved to : ' + './weights/critic' + file_name)
        return

    def load_models(self, weights_path):
        self.actor.load_state_dict(torch.load(weights_path[0]))
        self.critic.load_state_dict(torch.load(weights_path[1]))
        hard_copy(self.target_actor, self.actor)
        hard_copy(self.target_critic, self.critic)
        return

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        s1, a1, r1, s2 = self.memory.sample(self.batch_size)

        # ---------- Critic ----------
        a2 = self.target_actor(s2).detach()
        next_val = torch.squeeze(self.target_critic(s2, a2).detach())
        y_expected = r1 + self.gamma * next_val  # y_exp = r + gamma * Q'(s2, pi'(s2))
        y_predicted = torch.squeeze(self.critic(s1, a1))  # y_exp = Q(s1, a1)
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        # print('los_critic')
        # print(loss_critic)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------- Actor ----------
        pred_a1 = self.actor(s1)
        loss_actor = -1 * torch.sum(self.critic(s1, pred_a1))
        # print('loss_actor')
        # print(loss_actor)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        # Todo: Add verbose ?


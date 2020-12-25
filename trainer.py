import pyclick
import torch
import torch.nn.functional as F
import time
import pickle

import models
import utils.noise
import utils.screen
import utils.OCR
import utils.info_plot
import utils.schedule
from memory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DECAY = 0.9995

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

    def __init__(self, batch_size=5, lr=0.0001, tau=0.0001, gamma=0.999, load_weights=None, width=368, height=273):
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.actor = models.Actor().to(device)
        self.target_actor = models.Actor().to(device)
        self.critic = models.Critic().to(device)
        self.target_critic = models.Critic().to(device)
        print(self.actor)
        print(self.critic)
        if load_weights is not None:
            self.load_models(load_weights)
        else:
            hard_copy(self.target_actor, self.actor)
            hard_copy(self.target_critic, self.critic)

        self.noise = utils.noise.OrnsteinUhlenbeckActionNoise(mu=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device), sigma=0.25, theta=0.25, x0=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device))

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), self.lr/10.0, eps=0.00001)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), self.lr, eps=0.00001, weight_decay=0.0001)

        self.memory = ReplayMemory(3000)  # The larger the better because then the transitions have more chances to be uncorrelated

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        self.score_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_score2.pt')
        self.acc_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_acc2.pt')
        self.hc = pyclick.HumanClicker()

    def select_exploration_action(self, state, controls_state, episode_num=0):  # Check if the values are ok
        global t
        t += 1
        action = self.actor(state, controls_state).detach()
        if t % 159 == 0:
            print(action)
        new_action = action + DECAY**(episode_num +1) * self.noise.get_noise()
        return torch.clip(new_action, 0.0, 1.0)

    def select_exploitation_action(self, state, controls_state):
        action = self.target_actor(state, controls_state).detach()
        global t
        t += 1
        if t % 50 == 0:
            print(action)
        return action

    def save_model(self, file_name, num=0):
        torch.save(self.target_actor.state_dict(), './weights/actor' + file_name + str(num) + '.pt')
        print('Model saved to : ' + './weights/actor' + file_name + str(num) + '.pt')
        torch.save(self.target_critic.state_dict(), './weights/critic' + file_name + str(num) + '.pt')
        print('Model saved to : ' + './weights/critic' + file_name + str(num) + '.pt')
        return

    def load_models(self, weights_path):
        self.actor.load_state_dict(torch.load(weights_path[0]))
        print('Loaded actor weights from: ' + weights_path[0])
        self.critic.load_state_dict(torch.load(weights_path[1]))
        print('Loaded critic weights from :' + weights_path[1])
        hard_copy(self.target_actor, self.actor)
        hard_copy(self.target_critic, self.critic)
        return

    def optimize(self):
        if len(self.memory) < self.batch_size:
            time.sleep(0.02)
            return
        s1, a1, r1, s2, c_s1, c_s2 = self.memory.sample(self.batch_size)

        # ---------- Critic ----------
        a2 = self.target_actor(s2, c_s2).detach()  # (5, 4)
        next_val = torch.squeeze(self.target_critic(s2, c_s2, a2).detach())  # (5, 1) -> (5)
        y_expected = r1 + self.gamma * next_val  # y_exp = r + gamma * Q'(s2, pi'(s2))
        y_predicted = torch.squeeze(self.critic(s1, c_s1, a1))  # y_exp = Q(s1, a1)
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), 0.01)
        self.critic_optimizer.step()

        # ---------- Actor ----------
        pred_a1 = self.actor(s1, c_s1)
        loss_actor = -torch.mean(self.critic(s1, c_s1, pred_a1))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        #print(self.actor.fc6.weight.grad)
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.01)
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)


class QTrainer:
    def __init__(self, env, batch_size=32, lr=0.0001, gamma=0.999, initial_p=1.0, end_p=0.05, decay_p=2000000,
                 load_weights=None, load_memory=None, min_experience=25000, gradient_clipping_norm=10.0):
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma

        self.env = env

        self.q_network = models.QNetwork(action_dim=env.action_space.n, channels=env.stack_size).to(device)
        self.target_q_network = models.QNetwork(action_dim=env.action_space.n, channels=env.stack_size).to(device)
        print(self.q_network)
        if load_weights is not None:
            self.load_models(load_weights)
        self.optimizer = torch.optim.RMSprop(self.q_network.parameters(), self.lr, eps=0.01, alpha=0.95)
        self.scheduler = utils.schedule.LinearSchedule(decay_p, end_p, initial_p)

        if load_memory is None:
            self.memory = ReplayMemory(1000000) # TODO: Optimize memory
        else:
            self.memory = pickle.load(open(load_memory, 'rb'))
        self.min_experience = min_experience

        self.noise = utils.noise.OsuDiscreteNoise(mu=torch.tensor(0.5, device=device), sigma=torch.tensor(0.25, device=device), min_val=0.0, max_val=0.999)

        self.plotter = utils.info_plot.LivePlot(min_y=0, max_y=2.5, num_points=500, y_axis='Average loss')
        self.avg_reward_plotter = utils.info_plot.LivePlot(min_y=-10, max_y=250, window_x=1270, num_points=500, y_axis='Episode reward', x_axis='Number of episodes')
        self.running_loss = 0.0
        self.running_counter = 0

        self.gradient_clipping_norm = gradient_clipping_norm

        total_params = sum(p.numel() for p in self.q_network.parameters())
        print('Number of parameters: %d' % total_params)

    def select_action(self, state, controls_state):  # Greedy policy
        action = self.q_network(state, controls_state).detach()
        _, action = torch.max(action, 1)
        return action

    def random_action(self, x_discrete, y_discrete, click_dim=4):
        x = self.noise()  # Normal distribution mean=0.5, clipped in [0, 1[ & uniform distrib
        action = torch.tensor([int(x[0] * x_discrete) + x_discrete * int(
            x[1] * y_discrete) + x_discrete * y_discrete * int(x[2] * click_dim)], device=device)
        return action

    def save_model(self, file_name, num=0):
        torch.save(self.q_network.state_dict(), './weights/q_net_' + file_name + str(num) + '.pt')
        print('Model saved to : ' + './weights/q_net_' + file_name + str(num) + '.pt')

    def load_models(self, weights_path):
        self.q_network.load_state_dict(torch.load(weights_path))
        print('Loaded actor weights from: ' + weights_path)
        hard_copy(self.target_q_network, self.q_network)

    def optimize(self): # TODO: GO through the funciton and check. CHECK TERMINAL STATES thing. CLamp rewards and td error
        if len(self.memory) < self.min_experience:
            return
        s1, a1, r1, s2, c_s1, c_s2 = self.memory.sample(self.batch_size)
        s = self.q_network(s1, c_s1)
        state_action_values = torch.stack([s[i, a1[i]] for i in range(self.batch_size)])  # Get estimated Q(s1,a1)
        next_state_values = self.target_q_network(s2, c_s2).detach().max(1)[0]
        expected_state_action_values = r1 + self.gamma * next_state_values
        loss = torch.clamp(F.mse_loss(state_action_values, expected_state_action_values), 0, 1.0)
        self.running_loss += loss
        if self.running_counter % 200 == 0:
            self.plotter.step(self.running_loss/200)
            self.plotter.show()
            self.running_loss = 0.0
        self.running_counter += 1
        self.optimizer.zero_grad()
        loss.backward()
        '''
        for param in self.q_network.parameters():  # Todo: Benchmark to see whether this is faster than value_clip
            param.grad.data.clamp_(-0.1, 0.1)
        '''
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clipping_norm)

        self.optimizer.step()


if __name__ == '__main__':
    trainer = Trainer()
    while True:
        time.sleep(0.5)
        print(trainer.noise.get_noise())

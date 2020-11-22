import torch
import os
import PIL
import time
import random
import math
import pyclick
import pyautogui
import gc
import torch.nn.functional as F

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
import utils.network_updates
import models
from memory import ReplayMemory

BATCH_SIZE = 10
LEARNING_RATE = 0.001
GAMMA = 0.999
TAU = 0.001
EPS_START = 0.9 # ToDo: Tweak these hyperparameters
EPS_END = 0.05
EPS_DECAY  = 2000
TARGET_UPDATE = 10
# MAX_STEPS = 5000000
MAX_STEPS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


## Functions
def select_exploration_action(state, actor): # Check if the values are ok
    action = actor(state).detach()
    print(action)
    new_action = action.data + utils.noise.action_noise().to(device) * actor.width
    return new_action


mouse_state = [False, False] # 0 up, 1 down


def change_click(side, state): # left or right
    if side == 'left':
        if state[0]:
            pyautogui.mouseUp(button='left')
            state[0] = False
        else:
            pyautogui.mouseDown(button='left')
            state[0] = True
    else:
        if state[1]:
            pyautogui.mouseUp(button='right')
            state[1] = False
        else:
            pyautogui.mouseDown(button='right')
            state[1] = True


def perform_action(action, human_clicker, actor):
    t = torch.squeeze(action, 0).cpu()  # ToDo: Check for memory leak
    x, y = int(t[0].item()), int(t[1].item())
    print('(x, y) = (' + str(x) + ', ' + str(y) + ')')
    if t[2].item() < (actor.width/2):
        change_click('left', mouse_state)
    if t[3].item() < (actor.width/2):
        change_click('right', mouse_state)

    human_clicker.move((x, y), 0.1) # ToDo: Action bloquante, peut etre moyen de le threader
    return x, y


def get_reward(score, previous_score, x, y):
    ALPHA = 0.00001

    previous_x, previous_y = pyautogui.position()
    return (score - previous_score) - ALPHA * ((previous_x - x)**2 + (previous_y - y)**2)


def is_terminal_state(screen):
    return False  # ToDo: Code the function


def optimize(actor_optimizer, critic_optimizer, memory, actor, critic, target_actor, target_critic):
    if len(memory) < BATCH_SIZE:
        return
    s1, a1, r1, s2 = memory.sample(BATCH_SIZE)

    # ---------- Critic ----------
    a2 = target_actor(s2).detach()
    next_val = torch.squeeze(target_critic(s2, a2).detach())
    y_expected = r1 + GAMMA * next_val  # y_exp = r + gamma * Q'(s2, pi'(s2))
    y_predicted = torch.squeeze(critic(s1, a1))  # y_exp = Q(s1, a1)
    loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()

    # ---------- Actor ----------
    pred_a1 = actor(s1)
    loss_actor = -torch.sum(critic(s1, pred_a1))
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()

    utils.network_updates.soft_update(target_actor, actor, TAU)
    utils.network_updates.soft_update(target_critic, critic, TAU)
    # Todo: Add verbose ?


## Training
def train(episode_nb, learning_rate):
    process = utils.osu_routines.start_osu()

    # Networks & optimizers
    actor = models.Actor().to(device)
    target_actor = models.Actor().to(device)
    utils.network_updates.hard_copy(target_actor, actor)
    critic = models.Critic().to(device)
    target_critic = models.Critic().to(device)
    utils.network_updates.hard_copy(target_critic, critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), learning_rate)

    memory = ReplayMemory(10000)

    # Osu routine
    utils.osu_routines.move_to_songs(star=1)

    for i in range(episode_nb):
        screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        hc = pyclick.HumanClicker()

        utils.osu_routines.launch_random_beatmap()
        last_screen = utils.screen.get_screen(screen)
        current_screen = utils.screen.get_screen(screen)
        previous_score = 0
        state = torch.unsqueeze((current_screen - last_screen).permute(2, 0, 1), 0)

        for t in range(MAX_STEPS):
            action = select_exploration_action(state, actor)
            x, y = perform_action(action, hc, actor)
            score = utils.OCR.get_score(screen)
            reward = get_reward(score, previous_score, x, y)

            last_screen = current_screen
            current_screen = utils.screen.get_screen(screen)
            # current_screen = screen.get_latest_frame() ToDo: test this one, does it gets the last screenshot that has been done and is it in the buffer?
            #  ToDo: it could prevent small differences between the image used for score and the new state

            done = is_terminal_state(screen)
            if done:
                new_state = None
            else:
                new_state = torch.unsqueeze((current_screen - last_screen).permute(2, 0, 1), 0)
                memory.push(torch.squeeze(state, 0), torch.squeeze(action, 0), torch.tensor(reward).to(device), torch.squeeze(new_state, 0))

            optimize(actor_optimizer, critic_optimizer, memory, actor, critic, target_actor, target_critic)

            previous_score = score
            if done:
                break

        gc.collect() # Garbage collector at each episode
        # ToDo: Save / Load model
        # ToDo: Verbose

    screen.stop()
    utils.osu_routines.stop_osu(process)


if __name__ == '__main__':
    train(1, LEARNING_RATE)
    # process = utils.osu_routines.start_osu()
    # import utils.mouse_locator
    # utils.mouse_locator.mouse_locator()
    # utils.osu_routines.stop_osu(process)




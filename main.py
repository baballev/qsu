import torch
import os
import PIL
import time
import random
import math
import pyclick
import pyautogui
import gc

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
import utils.copy
import models
from memory import ReplayMemory

BATCH_SIZE = 10
LEARNING_RATE = 0.001
GAMMA = 0.999
EPS_START = 0.9 # ToDo: Tweak these parameters
EPS_END = 0.05
EPS_DECAY  = 2000
TARGET_UPDATE = 10
#MAX_STEPS = 5000000
MAX_STEPS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Functions


def select_exploration_action(state, actor): # Check if the values are ok
    action = actor(state).detach()
    new_action = action.data + utils.noise.action_noise() * actor.width
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
    t = action.cpu()  # ToDo: Check for memory leak
    x, y = t[0], t[1]
    if t[2] < 512:
        change_click('left', mouse_state)
    if t[3] < 512:
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
    s1, a1, r1, s2 = memory.sample(BATCH_SIZE)

    a2 = target_actor(s2).detach()
    next_val = 0 # ToDo


## Training
def train(episode_nb, learning_rate):
    process = utils.osu_routines.start_osu()

    actor = models.Actor().to(device)
    target_actor = models.Actor().to(device)
    utils.copy.hard_copy(target_actor, actor)
    critic = models.Critic().to(device)
    target_critic = models.Critic().to(device)
    utils.copy.hard_copy(target_critic, critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), learning_rate)

    memory = ReplayMemory(10000)

    # ToDo: Add osu_routine to choose a random beatmap

    for i in range(episode_nb):
        screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        hc = pyclick.HumanClicker()

        last_screen = utils.screen.get_screen(screen)
        current_screen = utils.screen.get_screen(screen)
        previous_score = 0
        state = current_screen - last_screen
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
                new_state = current_screen - last_screen
                memory.push(state, action, reward, new_state)

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





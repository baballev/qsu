import torch
import os
import PIL
import time
import random
import math
import pyclick
import pyautogui

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
import models
from memory import ReplayMemory

BATCH_SIZE = 10
GAMMA = 0.999
EPS_START = 0.9 # ToDo: Tweak these parameters
EPS_END = 0.05
EPS_DECAY  = 2000
TARGET_UPDATE = 10
MAX_STEPS = 5000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Functions


def select_exploration_action(state): # Check if the values are ok
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


def perform_action(action, human_clicker):
    t = action.cpu() #Check for memory leak
    if t[3] < 512:
        change_click('left', mouse_state)
    if t[4] < 512:
        change_click('right', mouse_state)

    human_clicker.move((t[0], t[1]), t[2]/(actor.width*5))


## Training
def train(episode_nb):
    process = utils.osu_routines.start_osu()

    actor = models.Actor().to(device)
    target_net = models.CNN().to(device)

    optimizer = torch.optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    # ToDo: Add osu_routine to choose a random beatmap

    for i in range(episode_nb):
        screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        hc = pyclick.HumanClicker()

        last_screen = utils.screen.get_screen(screen)
        current_screen = utils.screen.get_screen(screen)
        state = current_screen - last_screen
        for t in range(MAX_STEPS):
            action = select_exploration_action(state)
            # Action has been chosen, now 1) perform action 2) measure reward
            perform_action(action, hc)

            reward = utils.OCR.get_score(screen)



    screen.stop()
    utils.osu_routines.stop_osu(process)

if __name__ == '__main__':
    pass





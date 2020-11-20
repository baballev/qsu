import torch
import os
import PIL
import time

import utils.screen
import utils.osu_routines
from memory import ReplayMemory

BATCH_SIZE = 10
GAMMA = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory = ReplayMemory(10000)

process = utils.osu_routines.start_osu()
time.sleep(9)

screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
t = utils.screen.get_screen(screen)

utils.screen.save_screen(screen, os.curdir, 'test2.png')
time.sleep(5)

utils.osu_routines.stop_osu(process)


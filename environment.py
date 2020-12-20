import gym
from gym import spaces

import utils.screen


class OsuEnv(gym.Env):
    def __init__(self, discrete_width, discrete_height, height, width, stack_size):
        super(OsuEnv, self).__init__()

        self.action_space = spaces.Discrete(discrete_height * discrete_width * 4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, stack_size))

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        self.score_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_score2.pt')
        self.acc_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_acc2.pt')

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

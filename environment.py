import time
import gym
from gym import spaces

import utils.screen
import utils.osu_routines


class OsuEnv(gym.Env):
    def __init__(self, discrete_width, discrete_height, height, width, stack_size, star=2, beatmap_name=None, no_fail=False):
        super(OsuEnv, self).__init__()

        self.action_space = spaces.Discrete(discrete_height * discrete_width * 4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, stack_size))

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        self.score_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_score2.pt')
        self.acc_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_acc2.pt')
        self.process, self.window = utils.osu_routines.start_osu()

        self.star = star
        self.beatmap_name = beatmap_name
        self.no_fail = no_fail

        utils.osu_routines.move_to_songs(star=star)
        if beatmap_name is not None:
            utils.osu_routines.select_beatmap(beatmap_name)
        if no_fail:
            utils.osu_routines.enable_nofail()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def launch_episode(self):
        if self.beatmap_name is not None:
            utils.osu_routines.launch_selected_beatmap()
        else:
            utils.osu_routines.launch_random_beatmap()
        time.sleep(0.5)

    def change_star(self, star=2):
        pass # ToDo

    def stop(self):
        self.screen.stop()
        utils.osu_routines.stop_osu(self.process)

import time
import pyautogui
import pyclick
import gym
import torch
from random import randint
from threading import Thread
from gym import spaces

import utils.screen
import utils.osu_routines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def threaded_mouse_move(x, y, t, human_clicker, curve):
    human_clicker.move((x, y), t, humanCurve=curve)
    return


class OsuEnv(gym.Env):
    def __init__(self, discrete_width, discrete_height, discrete_factor, height, width, stack_size, star=2, beatmap_name=None, no_fail=False, skip_pixels=4):
        super(OsuEnv, self).__init__()
        self.discrete_width = discrete_width
        self.discrete_height = discrete_height
        self.height = height
        self.width = width
        self.stack_size = stack_size
        self.discrete_factor = discrete_factor
        print(self.discrete_factor)

        self.action_space = spaces.Discrete(discrete_height * discrete_width * 4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, stack_size))

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        self.score_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_score2.pt')
        self.acc_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_acc2.pt')
        self.process, self.window = utils.osu_routines.start_osu()
        self.hc = pyclick.HumanClicker()

        self.star = star
        self.beatmap_name = beatmap_name
        self.no_fail = no_fail
        self.skip_pixels = skip_pixels
        self.first = True

        self.history = None
        self.previous_score = None
        self.previous_acc = None
        self.thread = None

        utils.osu_routines.move_to_songs(star=star)
        if beatmap_name is not None:
            utils.osu_routines.select_beatmap(beatmap_name)
        if no_fail:
            utils.osu_routines.enable_nofail()

    def step(self, action, step):
        new_controls_state = self.perform_discrete_action(action, self.hc)
        for i in range(len(self.history)-1):
            self.history[i] = self.history[i+1]
        # TODO: maybe try threading every actions that can be if i need to win some time.
        self.history[-1] = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels).sum(0, keepdim=True) / 3.0

        score, acc = utils.OCR.get_score_acc(self.screen, self.score_ocr, self.acc_ocr, self.window)
        if (step < 15 and score == -1) or (score - self.previous_score > 5 * (self.previous_score + 100)):
            score = self.previous_score
        if step < 15 and acc == -1:
            acc = self.previous_acc
        done = (score == -1)
        reward = self.get_reward(score, acc, step)  # TODO: remove step?
        self.previous_acc = acc
        self.previous_score = score

        return self.history.unsqueeze(0), new_controls_state, reward, done

    def reset(self):
        if not self.first:
            utils.osu_routines.return_to_beatmap()
        self.first = False
        pyautogui.mouseUp(button='right')
        pyautogui.mouseUp(button='left')
        state = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels).sum(0, keepdim=True) / 3.0
        self.history = torch.cat([state for _ in range(self.stack_size-1)])
        state = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels).sum(0, keepdim=True) / 3.0
        self.history = torch.cat((self.history, state), 0)
        self.previous_score = torch.tensor(0.0, device=device)
        self.previous_acc = torch.tensor(100.0, device=device)
        return torch.tensor([[0.5, 0.5, 0.0, 0.0]], device=device), self.history.unsqueeze(0)

    def render(self, mode='human', close=False):
        pass  # TODO

    def launch_episode(self):
        if self.beatmap_name is not None:
            utils.osu_routines.launch_selected_beatmap()
        else:
            utils.osu_routines.launch_random_beatmap()
        time.sleep(0.5)

    def change_star(self, star=2):
        pass # ToDo

    def stop(self):
        self.reset()
        self.screen.stop()
        utils.osu_routines.stop_osu(self.process)

    def get_reward(self, score, acc, step): # TODO
        if acc > self.previous_acc:
            bonus = torch.tensor(1.0, device=device)
        elif acc < self.previous_acc:
            bonus = torch.tensor(-1.0, device=device)
        else:
            if acc < 0.2 + 0.1 * (step // 500):
                bonus = torch.tensor(-0.3, device=device)
            else:
                bonus = torch.tensor(0.0, device=device)
        return torch.clamp(torch.log10(max((score - self.previous_score),
                                           torch.tensor(1.0, device=device))) + bonus, -1, 1)

    def perform_discrete_action(self, action, human_clicker, dt=0.08):
        click, xy = divmod(action.item(), self.discrete_width * self.discrete_height)
        y_disc, x_disc = divmod(xy, self.discrete_width)
        if click == 0:
            pyautogui.mouseUp(button="left")
            pyautogui.mouseUp(button="right")
            left, right = 0.0, 0.0
        elif click == 1:
            pyautogui.mouseDown(button="left")
            pyautogui.mouseUp(button="right")
            left, right = 1.0, 0.0
        elif click == 2:
            pyautogui.mouseUp(button="left")
            pyautogui.mouseDown(button="right")
            left, right = 0.0, 1.0
        else:  # ToDo: I could use this to make a delayed click instead?
            pyautogui.mouseDown(button="left")
            pyautogui.mouseDown(button="right")
            left, right = 1.0, 1.0
        x = x_disc * self.discrete_factor + 145  # + randint(-DISCRETE_FACTOR // 3, DISCRETE_FACTOR // 3)
        y = y_disc * self.discrete_factor + 54 + 26  # + randint(-DISCRETE_FACTOR // 3, DISCRETE_FACTOR // 3)

        curve = pyclick.HumanCurve(pyautogui.position(), (x, y), targetPoints=15)  # TODO: ACTION BLOQUANTE?

        if self.thread is not None:
            self.thread.join()
        self.thread = Thread(target=threaded_mouse_move, args=(x, y, dt, human_clicker, curve)) # TODO: TRY TO UN-THREAD the action
        self.thread.start()
        return torch.tensor([[left, right, x / self.width, y / self.height]], device=device)




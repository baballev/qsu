import time
import pyautogui
import pyclick
import gym
import torch
import win32api, win32con
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
    def __init__(self, discrete_width, discrete_height, discrete_factor, height, width, stack_size, star=2, beatmap_name=None, no_fail=False, skip_pixels=4, human_off_policy=False):
        super(OsuEnv, self).__init__()
        self.discrete_width = discrete_width
        self.discrete_height = discrete_height
        self.height = height
        self.width = width
        self.stack_size = stack_size
        self.discrete_factor = discrete_factor

        self.human_off_policy = human_off_policy

        self.action_space = spaces.Discrete(discrete_height * discrete_width * 4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(height, width, stack_size))

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
        self.score_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_score2.pt').to(device)
        self.acc_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_acc2.pt').to(device)
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

    def step(self, action, steps):
        new_controls_state = self.perform_discrete_action(action, self.hc)
        for i in range(len(self.history)-1):
            self.history[i] = self.history[i+1]
        # TODO: maybe try threading every actions that can be if i need to win some time.
        time.sleep(0.025)
        th = Thread(target=self.threaded_screen_fetch)
        th.start()
        score, acc = utils.OCR.get_score_acc(self.screen, self.score_ocr, self.acc_ocr, self.window)
        if (steps < 15 and score == -1) or (score - self.previous_score > 5 * (self.previous_score + 100)):
            score = self.previous_score
        if steps < 15 and acc == -1:
            acc = self.previous_acc
        done = (score == -1)
        th.join()
        if self.history[-1, 1, 1] > 0.0834 and steps > 25:
            done = True
            reward = torch.tensor(-1.0, device=device)
        else:
            reward = self.get_reward(score, acc)  # TODO: remove step?
        self.previous_acc = acc
        self.previous_score = score

        return self.history.unsqueeze(0), new_controls_state, reward, done

    def reset(self, reward):
        if not self.first:
            if reward != -1.0:
                utils.osu_routines.return_to_beatmap()
            else:
                if self.beatmap_name is not None:
                    utils.osu_routines.restart()
                else:
                    utils.osu_routines.return_to_beatmap2()
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

    def observe(self, steps, dt=1/(9.0*3)):
        time.sleep(dt)
        x, y = pyautogui.position()
        left, right = (win32api.GetAsyncKeyState(ord('C')) <= -32767), (win32api.GetAsyncKeyState(ord('V')) <= -32767)
        controls_state = torch.tensor([[left, right, x / self.width, y / self.height]], device=device)
        for i in range(len(self.history)-1):
            self.history[i] = self.history[i+1]
        # TODO: maybe try threading every actions that can be if i need to win some time.
        self.history[-1] = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels).sum(0, keepdim=True) / 3.0
        score, acc = utils.OCR.get_score_acc(self.screen, self.score_ocr, self.acc_ocr, self.window)
        if (steps < 15 and score == -1) or (score - self.previous_score > 5 * (self.previous_score + 100)):
            score = self.previous_score
        if steps < 15 and acc == -1:
            acc = self.previous_acc
        done = (score == -1)
        if self.history[-1, 1, 1] > 0.0834 and steps > 25:
            done = True
            reward = torch.tensor(-1.0, device=device)
        else:
            reward = self.get_reward(score, acc)  # TODO: remove step?
        self.previous_acc = acc
        self.previous_score = score

        if left and right: v = 3
        elif right: v = 2
        elif left: v = 1
        else: v = 0

        action = torch.tensor(v * (self.discrete_width * self.discrete_height) + int((y/self.height) * self.discrete_height) * self.discrete_width + int((x / self.width ) * self.discrete_width), device=device)
        return action, self.history.unsqueeze(0), controls_state, reward, done

    def render(self, mode='human', close=False):
        pass  # TODO

    def launch_episode(self, reward):
        if reward != -1:
            if self.beatmap_name is not None:
                utils.osu_routines.launch_selected_beatmap()
            else:
                utils.osu_routines.launch_random_beatmap()
        else:
            if self.beatmap_name is None:
                utils.osu_routines.launch_random_beatmap()
        time.sleep(0.3)

    def change_star(self, star=2):
        pass # ToDo

    def stop(self):
        self.reset(0.5)
        self.screen.stop()
        utils.osu_routines.stop_osu(self.process)

    def get_reward(self, score, acc):
        if acc > self.previous_acc:
            bonus = torch.tensor(0.3, device=device)
        elif acc < self.previous_acc:
            bonus = torch.tensor(-0.3, device=device)
        else:
            bonus = torch.tensor(0.1, device=device)
        return torch.clamp(0.1*torch.log10(max((score - self.previous_score),
                                           torch.tensor(1.0, device=device))) + bonus, -1, 1)

    def perform_discrete_action(self, action, human_clicker, dt=0.05):
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

        curve = pyclick.HumanCurve(pyautogui.position(), (x, y), targetPoints=10)
        if self.thread is not None:
            self.thread.join()
        self.thread = Thread(target=threaded_mouse_move, args=(x, y, dt, human_clicker, curve))
        self.thread.start()
        return torch.tensor([[left, right, x / self.width, y / self.height]], device=device)

    def threaded_screen_fetch(self):
        self.history[-1] = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels).sum(0, keepdim=True) / 3.0



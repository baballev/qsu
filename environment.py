import math
import time
import pyautogui
import pyclick
import gym
import torch
import win32api, win32con
from threading import Thread
from gym import spaces

import utils.screen
import utils.osu_routines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available


def threaded_mouse_move(x, y, t, human_clicker, curve):  # Threaded function handling mouse movement
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

    def observe(self, steps, dt=1/(9.0*3)):  # Obsoltete
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
        pass  # ToDo

    def stop(self):
        self.reset(0.5)
        self.screen.stop()
        utils.osu_routines.stop_osu(self.process)

    def get_reward(self, score, acc):  # ToDo: Rethink this reward function with and without nofail
        if acc > self.previous_acc:    # ToDo: CLip it in [-1, 1] as well as TD-error in the optimization func, not here
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


class ManiaEnv(gym.Env):  # Environment use to play the osu! mania mode (only keyboard, no mouse move involved)
    def __init__(self, width=1024, height=600, stack_size=4, star=4, beatmap_name=None, skip_pixels=4, num_actions=128,
                 no_fail=False, acc_threshold=1.0):
        super(ManiaEnv, self).__init__()
        self.stack_size = stack_size  # Length of history, number of last frames to pass as state
        self.width = width
        self.height = height
        self.skip_pixels = skip_pixels
        self.region = (140//skip_pixels, 520//skip_pixels)

        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(height//skip_pixels, width//skip_pixels, stack_size))

        self.screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")  # Object fetching game screen
        self.score_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_score2.pt').to(device)  # OCR model to read score
        self.acc_ocr = utils.OCR.init_OCR('./weights/OCR/OCR_acc2.pt').to(device)  # OCR model to read accuracy
        self.process, self.window = utils.osu_routines.start_osu()  # Osu! process + window object after launching game

        self.key_dict = {'d': 0x44, 'f': 0x46, 'j': 0x4A, 'k': 0x4B, 'l': 0x4C, 's': 0x53, ' ': 0x20}  # Keyboard encoding

        self.star = star  # Difficulty level of the songs played by the agent, integer from 1 to 10
        self.beatmap_name = beatmap_name  # If the agent plays a specific map, he will search it using this name
        self.no_fail = no_fail  # True if no fail mode is active, False otherwise
        self.skip_pixels = skip_pixels  # Original game screen width, height --> width//skip_pixels, height//skip_pixels
        self.acc_threshold = acc_threshold  # Used for reward, under this threshold (percent) agent gets negative reward

        self.previous_score = None
        self.previous_acc = None
        self.history = None  # Tracking the last 4 frames and send it as state to the neural network

        self.steps = 0  # Number of steps into episode tracker
        self.episode_counter = 0
        self.last_action = None

        utils.osu_routines.move_to_songs(star=star)
        if beatmap_name is not None:
            utils.osu_routines.select_beatmap(beatmap_name)
        if self.no_fail:
            utils.osu_routines.enable_nofail()

    def perform_actions(self, control):  # Control in [0, 127=2**7-1] or [0, 15]
        n = math.floor(math.log2(self.action_space.n))  # Number of buttons ('7K' or '4K' usually)
        key_encoding = [0 for _ in range(n)]
        q, r = divmod(control, 2**(n - 1))
        i = 0
        key_encoding[i] = q
        for _ in range(n-1):
            i += 1
            q, r = divmod(r, 2**(n - (i+1)))
            key_encoding[i] = q
        if n == 7:  # 7K mode -> S, D, F, SPACE, J, K, L
            for i, key in enumerate(self.key_dict.keys()):
                if key_encoding[i]:
                    win32api.keybd_event(self.key_dict[key], 0, 0, 0)  # Press keyboard button
                else:
                    win32api.keybd_event(self.key_dict[key], 0, win32con.KEYEVENTF_KEYUP, 0)  # Release keyboard button
        if n == 4:  # 4K mode -> D, F, J, K
            for i, key in enumerate(self.key_dict.keys()):
                if i >= 4:
                    break
                if key_encoding[i]:
                    win32api.keybd_event(self.key_dict[key], 0, 0, 0)  # Press keyboard button
                else:
                    win32api.keybd_event(self.key_dict[key], 0, win32con.KEYEVENTF_KEYUP, 0)  # Release keyboard button

    def launch_episode(self, reward):  # Combination of routines to launch the beatmap or restart it if failed
        if reward != -1:
            if self.beatmap_name is not None:
                utils.osu_routines.launch_selected_beatmap()
            else:
                utils.osu_routines.launch_random_beatmap()
        else:
            if self.beatmap_name is None:
                utils.osu_routines.launch_random_beatmap()
        time.sleep(0.3)

    def reset(self, reward=0.0):
        if self.steps != 0:  # If this is not the beginning of training
            if reward != -1.0:  # When no_fail = False, if the agent fails the map he gets a -1.0 reward
                utils.osu_routines.return_to_beatmap()  # If he did not get -1.0, he finished the map so return to menu
            else:
                if self.beatmap_name is not None:
                    utils.osu_routines.restart()  # Else, he failed the map so he can simply restart it on the fail screen
                else:
                    utils.osu_routines.return_to_beatmap2()  # Or he can choose another random beatmap by going to menu
            self.episode_counter += 1
        self.steps = 0
        for key in self.key_dict.keys():
            win32api.keybd_event(self.key_dict[key], 0, win32con.KEYEVENTF_KEYUP, 0)
        state = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels)[:, :, self.region[0]:self.region[1]].sum(0, keepdim=True) / 3.0
        self.history = torch.cat([state for _ in range(self.stack_size - 1)])
        state = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels)[:, :, self.region[0]:self.region[1]].sum(0, keepdim=True) / 3.0
        self.history = torch.cat((self.history, state), 0)
        self.previous_score = torch.tensor(0.0, device=device)
        self.previous_acc = torch.tensor(100.0, device=device)
        return self.history.unsqueeze(0)

    def get_reward(self, score, acc):
        if acc > self.previous_acc:  # If the accuracy increased, positive reward
            bonus = torch.tensor(0.3, device=device)
        elif acc < self.previous_acc:  # If accuracy decreased, negative reward
            bonus = torch.tensor(-0.2, device=device)
        else:  # If accuracy remained the same (no change)
            if self.no_fail:  # If the no fail mode is enabled
                if acc < self.acc_threshold:  # If accuracy is very low (like 0.0%), negative reward
                    bonus = torch.tensor(-0.1, device=device)
                else:  # Else, it probably means there is no reward obtainable so just give 0 reward
                    bonus = torch.tensor(0.0, device=device)
            else:
                bonus = torch.tensor(0.1, device=device)  # Bonus for surviving and not failing map
        return 0.1 * torch.log10(torch.tensor(max((score - self.previous_score), 1.0), device=device)) + bonus

    def step(self, action):  # TODO: Important note: problem with d3dshot, need to make a pull request and fix issue because taking 2
        self.steps += 1      # TODO: screenshots consecutively doesnt work, there need to be a small sleep
        self.perform_actions(action)
        self.last_action = action
        for i in range(len(self.history)-1):
            self.history[i] = self.history[i+1]  # Update history
        time.sleep(0.015)  # Frequency play regulator, wait a bit after action has been performed before observing
        self.history[-1] = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels)[:, :, self.region[0]:self.region[1]].sum(0, keepdim=True) / 3.0
        score, acc = utils.OCR.get_score_acc(self.screen, self.score_ocr, self.acc_ocr, self.window)

        if (self.steps < 25 and score == -1) or (score - self.previous_score > 5 * (self.previous_score + 100)):
            score = self.previous_score  # If the OCR failed to read the score, set it to the previous score
        if self.steps < 25 and acc == -1:  # Score/Accuracy Reading problems often occur during the first 25 first frames
            acc = self.previous_acc  # If the OCR failed to read the accuracy at the beginning of the map, keep it at the same value

        done = (score == -1)  # When the beatmap is done, the screen becomes black, OCR detects it and returns -1
        rew = self.get_reward(score, acc)
        if not self.no_fail:
            if self.history[-1, 1, 1] > 0.0834 and self.steps > 30:  # If left hand corner pixel becomes too red -> agent failed the map
                done = True
                rew = torch.tensor(-1.0, device=device)  # Negative reward for failing the map
        self.previous_acc = acc
        self.previous_score = score

        return self.history.unsqueeze(0), rew, done


    def threaded_screen_fetch(self):
        self.history[-1] = utils.screen.get_game_screen(self.screen, skip_pixels=self.skip_pixels)[:, :, self.region[0]:self.region[1]].sum(0, keepdim=True) / 3.0

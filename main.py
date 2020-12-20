import math
from random import random, randint
import torch
import time
import pyautogui
import pyclick
import gc
from threading import Thread

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
import utils.info_plot
import environment
from trainer import QTrainer

torch.cuda.empty_cache()
torch.set_printoptions(sci_mode=False)

pyautogui.MINIMUM_DURATION = 0.0
pyautogui.MINIMUM_SLEEP = 0.0
pyautogui.PAUSE = 0.0

BATCH_SIZE = 20
LEARNING_RATE = 0.00001
GAMMA = 0.999
MAX_STEPS = 25000
WIDTH = 878
HEIGHT = 600
STACK_SIZE = 4

EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 300000
TARGET_UPDATE = 10

DISCRETE_FACTOR = 10
X_DISCRETE = 685 // DISCRETE_FACTOR + 1
Y_DISCRETE = (560 - 54) // DISCRETE_FACTOR + 1

PIXEL_SKIP = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Functions
def threaded_mouse_move(x, y, t, human_clicker, curve):
    human_clicker.move((x, y), t, humanCurve=curve)
    return


def get_reward(score, previous_score, acc, previous_acc, step):
    if acc > previous_acc:
        bonus = torch.tensor(1.0, device=device)
    elif acc < previous_acc:
        bonus = torch.tensor(-1.0, device=device)
    else:
        if acc < 0.2 + 0.1 * (step // 500):
            bonus = torch.tensor(-0.3, device=device)
        else:
            bonus = torch.tensor(0.0, device=device)
    return torch.clamp(torch.log10(max((score - previous_score),
                           torch.tensor(1.0, device=device))) + bonus, -1, 1)


def perform_discrete_action(action, human_clicker, frequency, thre):
    click, xy = divmod(action.item(), X_DISCRETE * Y_DISCRETE)
    x_disc, y_disc = divmod(xy, X_DISCRETE)
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
    x = x_disc * DISCRETE_FACTOR + 145 + randint(-DISCRETE_FACTOR // 3, DISCRETE_FACTOR // 3)
    y = y_disc * DISCRETE_FACTOR + 54 + 26 + randint(-DISCRETE_FACTOR // 3, DISCRETE_FACTOR // 3)

    curve = pyclick.HumanCurve(pyautogui.position(), (x, y), targetPoints=25 - frequency)
    if thre is not None:
        thre.join()
    thr = Thread(target=threaded_mouse_move, args=(x, y, 1 / frequency, human_clicker, curve))
    thr.start()
    return torch.tensor([[left, right, x / WIDTH, y / HEIGHT]], device=device), thr


def trainQNetwork(episode_nb, learning_rate, batch_size=BATCH_SIZE, load_weights=None, save_name='tests',
                  beatmap_name=None, star=1, frequency=10, evaluation=False):
    training_steps = 0
    print('Discretized x: ' + str(X_DISCRETE))
    print('Discretized y: ' + str(Y_DISCRETE))
    print('Action dim: ' + str(X_DISCRETE * Y_DISCRETE * 4))
    print('X_MAX = ' + str(145 + (X_DISCRETE - 1) * DISCRETE_FACTOR))
    print('YMAX = ' + str(54 + 26 + (Y_DISCRETE - 1) * DISCRETE_FACTOR))
    if evaluation:
        learning_rate = 0.0

    env = environment.OsuEnv(X_DISCRETE, Y_DISCRETE, WIDTH, HEIGHT, STACK_SIZE, star=star, beatmap_name=beatmap_name, no_fail=True)
    q_trainer = QTrainer(env, batch_size=batch_size, lr=learning_rate, load_weights=load_weights, skip_pixels=PIXEL_SKIP)

    episodes_reward = 0.0
    episode_average_reward = 0.0
    c = 0
    k = 0
    for i in range(episode_nb):
        env.launch_episode()
        previous_score = torch.tensor(0.0, device=device)
        previous_acc = torch.tensor(100.0, device=device)
        controls_state = torch.tensor([[0.5, 0.5, 0.0, 0.0]], device=device)
        state = utils.screen.get_game_screen(env.screen, skip_pixels=PIXEL_SKIP).unsqueeze_(0).sum(1, keepdim=True) / 3.0
        thread, thr, thready_mercury = None, None, None
        start = time.time()
        for step in range(MAX_STEPS):
            k += 1
            with torch.no_grad():
                if evaluation:  # Choose greedy policy if tests
                    action = q_trainer.select_action(state, controls_state)
                else:  # Else choose an epsilon greedy policy with decaying epsilon
                    sample = random()
                    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * training_steps / EPS_DECAY)
                    training_steps += 1
                    if sample > eps_threshold:
                        action = q_trainer.select_action(state, controls_state)
                    else:
                        x = q_trainer.noise()  # Normal distribution mean=0.5, clipped in [0, 1[
                        action = torch.tensor([int(x[0] * X_DISCRETE) + X_DISCRETE * int(
                            x[1] * Y_DISCRETE) + X_DISCRETE * Y_DISCRETE * int(x[2] * 4)], device=device)

                new_controls_state, thr = perform_discrete_action(action, q_trainer.hc, frequency, thr)
                new_state = utils.screen.get_game_screen(env.screen, skip_pixels=PIXEL_SKIP).unsqueeze_(0).sum(1, keepdim=True) / 3.0
                score, acc = utils.OCR.get_score_acc(env.screen, env.score_ocr, env.acc_ocr, env.window)

                if (step < 15 and score == -1) or (score - previous_score > 5 * (previous_score + 100)):
                    score = previous_score
                if step < 15 and acc == -1:
                    acc = previous_acc
                reward = get_reward(score, previous_score, acc, previous_acc, step)
                done = (score == -1)
                if done:
                    new_state = None
                else:
                    th = Thread(target=q_trainer.memory.push,
                                args=(state, action, reward, new_state, controls_state, new_controls_state))
                    th.start()

            if thread is not None:
                thread.join()
            thread = Thread(target=q_trainer.optimize)
            thread.start()

            previous_score = score
            previous_acc = acc
            state = new_state
            controls_state = new_controls_state

            episode_average_reward += reward
            if k % 200 == 0:
                tmp = episode_average_reward / 200
                q_trainer.avg_reward_plotter.step(tmp)
                q_trainer.avg_reward_plotter.show()
                episodes_reward += tmp
                c += 1
                episode_average_reward = 0.0
            if done:
                break
        end = time.time()
        delta_t = end - start
        print(str(step) + ' time steps in ' + str(delta_t) + ' s.')
        print(str(step / delta_t) + ' time_steps per second.')
        gc.collect()

        q_trainer.avg_reward_plotter.fit()
        q_trainer.avg_reward_plotter.show()

        if i % 50 == 0:
            q_trainer.plotter.fig.savefig('average_loss' + str(i) + '.png')
            q_trainer.avg_reward_plotter.fig.savefig('average_reward' + str(i) + '.png')

        if i % TARGET_UPDATE == 0:
            q_trainer.target_q_network.load_state_dict(q_trainer.q_network.state_dict())

        if i % 10 == 0 and i > 0:
            print('Mean reward over last 10 episodes: ')
            print(episodes_reward / c)

            c = 0
            episodes_reward = 0.0
            if beatmap_name is not None:
                tmp = beatmap_name + save_name
            else:
                tmp = save_name
            q_trainer.save_model(tmp, num=i)

        utils.osu_routines.return_to_beatmap()
        pyautogui.mouseUp(button='right')
        pyautogui.mouseUp(button='left')

    if (episode_nb - 1) % 5 != 0:
        print('Mean reward over last episodes: ')
        print(episodes_reward / c)
        if beatmap_name is not None:
            tmp = beatmap_name + save_name
        else:
            tmp = save_name
        q_trainer.save_model(tmp, num=episode_nb - 1)
        q_trainer.plotter.fig.savefig('average_loss' + str(episode_nb-1) + '.png')
        q_trainer.avg_reward_plotter.fig.savefig('average_reward' + str(episode_nb-1) + '.png')

    env.stop()


if __name__ == '__main__':
    weights_path = './weights/q_net_fubuki guysReboot_12-12-2020-210.pt'
    save_name = 'reboot_13-12-2020-'
    trainQNetwork(1, LEARNING_RATE, evaluation=False, load_weights=None, beatmap_name="rampage", star=2,
                  save_name=save_name, batch_size=BATCH_SIZE)

import math
from random import random, randint
import torch
import time
import pyautogui
import gc
from threading import Thread

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
from trainer import Trainer, QTrainer

torch.cuda.empty_cache()
torch.set_printoptions(sci_mode=False)

BATCH_SIZE = 10
LEARNING_RATE = 0.00001
GAMMA = 0.999
TAU = 0.0001
MAX_STEPS = 25000
WIDTH = 735
HEIGHT = 546

EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 125000
TARGET_UPDATE = 5

DISCRETE_FACTOR = 10
X_DISCRETE = 685//DISCRETE_FACTOR + 1
Y_DISCRETE = (560-54)//DISCRETE_FACTOR + 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


## Functions
def threaded_mouse_move(x, y, t, human_clicker):
    human_clicker.move((x, y), t)
    return


def perform_action(action, human_clicker):
    x, y = action[0][0],  action[0][1]
    if action[0][2] > 0.5:
        pyautogui.mouseDown(button='left')
        left = 1.0
    else:
        pyautogui.mouseUp(button='left')
        left = 0.0
    if action[0][3] > 0.5:
        pyautogui.mouseDown(button='right')
        right = 1.0
    else:
        pyautogui.mouseUp(button='right')
        right = 0.0
    if y*HEIGHT < 54:
        thread = Thread(target=threaded_mouse_move, args=(min(int(x*WIDTH)+145, 830), 54+26, 0.08, human_clicker))
        thread.start()
    elif y*HEIGHT > 560:
        thread = Thread(target=threaded_mouse_move, args=(min(int(x*WIDTH)+145, 830), 560+26, 0.08, human_clicker))
        thread.start()
    elif x*WIDTH > 685:
        thread = Thread(target=threaded_mouse_move, args=(685+145, int(y*HEIGHT)+26, 0.08, human_clicker))
        thread.start()
    else:
        thread = Thread(target=threaded_mouse_move, args=(max(int(x*WIDTH)+145, 145), int(y*HEIGHT)+26, 0.08, human_clicker))
        thread.start()
    return torch.tensor([[left, right]], device=device)


def get_reward(score, previous_score, acc, previous_acc, step):
    if acc > previous_acc:
        bonus = torch.tensor(1.0, device=device)
    elif acc < previous_acc:
        bonus = torch.tensor(-0.3, device=device)
    else:
        if acc < 0.2 + 0.1 * (step//500):
            bonus = torch.tensor(-0.3, device=device)
        else:
            bonus = torch.tensor(0.0, device=device)
    return torch.log10(max((score - previous_score), torch.tensor(1.0, device=device))) + bonus  # - 0.005 * ((x - 0.5)**2 + (y - 0.5)**2)


## Training
def trainDDPG(episode_nb, learning_rate, batch_size=BATCH_SIZE, load_weights=None, save_name='tests', beatmap_name=None, star=1, frequency=9.0):
    # Osu routine
    process, wndw = utils.osu_routines.start_osu()
    utils.osu_routines.move_to_songs(star=star)
    if beatmap_name is not None:
        utils.osu_routines.select_beatmap(beatmap_name)
    utils.osu_routines.enable_nofail()
    trainer = Trainer(load_weights=load_weights, lr=learning_rate, batch_size=batch_size, tau=TAU, gamma=GAMMA)
    episodes_reward = 0
    episode_average_reward = 0.0
    c = 0
    k = 0
    for i in range(episode_nb):
        #best = -5.0
        utils.osu_routines.launch_random_beatmap()
        #previous_screen = utils.screen.get_game_screen(trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0
        previous_score = torch.tensor(0.0, device=device)
        previous_acc = torch.tensor(100.0, device=device)
        controls_state = torch.tensor([[0.5, 0.5, 0.0, 0.0]], device=device)
        current_screen = utils.screen.get_game_screen(trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0
        state = current_screen #- previous_screen
        start = time.time()
        thread = None
        #logger = open('./benchmark/log.txt', 'w+')
        for step in range(MAX_STEPS):
            step_time_prev = time.time()
            k += 1
            with torch.no_grad():
                action = trainer.select_exploration_action(state, controls_state, i)
                #action = trainer.select_exploitation_action(state, controls_state)
                new_controls_state = perform_action(action, trainer.hc)
                #previous_screen = current_screen
                current_screen = (utils.screen.get_game_screen(trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0)
                score, acc = utils.OCR.get_score_acc(trainer.screen, trainer.score_ocr, trainer.acc_ocr, wndw)
                new_x, new_y = pyautogui.position()
                new_controls_state = torch.cat((torch.tensor([[new_x/WIDTH, new_y/HEIGHT]], dtype=torch.float32, device=device), new_controls_state), 1)
                if (step < 15 and score == -1) or (score - previous_score > 5*(previous_score+100)):
                    score = previous_score
                if step < 15 and acc == -1:
                    acc = previous_acc
                reward = get_reward(score, previous_score, acc, previous_acc, step)
                #print(reward)
                done = (score == -1)
                if done:
                    new_state = None
                else:
                    new_state = current_screen# - previous_screen
                    #t = torch.squeeze(trainer.critic(state, controls_state, action))
                    #print(t)
                    #    torchvision.transforms.ToPILImage()(torch.squeeze(state)).save('./benchmark/' + str(step) + '__' '''+ str(t.item())''' + '_.png')
                    #    logger.write(str(step) + '___' + str(t.item()) + '___' + str(controls_state) + '___' + str(action) + '\n')
                    th = Thread(target=trainer.memory.push, args=(state, action, reward, new_state, controls_state, new_controls_state))
                    th.start()
                    #trainer.memory.push(state, action, reward, new_state, controls_state, new_controls_state)

            if thread is not None:
                thread.join()
            thread = Thread(target=trainer.optimize)
            thread.start()
            #trainer.optimize()

            previous_score = score
            previous_acc = acc
            state = new_state
            controls_state = new_controls_state
            episode_average_reward += reward
            if k % 1000 == 0:
                print('Reward average over last 1000 steps: ')
                tmp = episode_average_reward/1000
                print(tmp)
                episodes_reward += tmp
                c += 1
                episode_average_reward = 0.0

            if done:
                break

            step_time_curr = time.time()
            if step_time_curr - step_time_prev < 1/frequency:
                dt = 1/frequency - (step_time_curr - step_time_prev)
                time.sleep(dt/1.015)

        end = time.time()
        delta_t = end - start
        #logger.close()
        print(str(step) + ' time steps in ' + str(delta_t) + ' s.')
        print(str(step/delta_t) + ' time_steps per second.')
        gc.collect()  # Garbage collector at each episode

        if i % 20 == 0 and i > 0:
            print('Mean reward over last 20 episodes: ')
            print(episodes_reward/c)
            '''if episodes_reward/c > best:
                trainer(save_name + 'best', num=0)
            '''
            c = 0
            episodes_reward = 0.0
            if beatmap_name is not None:
                tmp = beatmap_name + save_name
            else:
                tmp = save_name
            trainer.save_model(tmp, num=i)

        utils.osu_routines.return_to_beatmap()
        trainer.noise.reset()
        pyautogui.mouseUp(button='right')
        pyautogui.mouseUp(button='left')

    if (episode_nb - 1) % 15 != 0:
        print('Mean reward over last episodes: ')
        print(episodes_reward/c)
        if beatmap_name is not None:
            tmp = beatmap_name + save_name
        else:
            tmp = save_name
        trainer.save_model(tmp, num=episode_nb-1)

    trainer.screen.stop()
    utils.osu_routines.stop_osu(process)


def perform_discrete_action(action, human_clicker, thread):
    click, xy = divmod(action.item(), X_DISCRETE*Y_DISCRETE)
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
    x = x_disc * DISCRETE_FACTOR + 145 + randint(-DISCRETE_FACTOR//2, DISCRETE_FACTOR//2)
    y = y_disc * DISCRETE_FACTOR + 54 + 26 + randint(-DISCRETE_FACTOR//2, DISCRETE_FACTOR//2)

    th = Thread(target=threaded_mouse_move, args=(x, y, 0.1, human_clicker))
    if thread is not None:
        thread.join()
    th.start()
    return torch.tensor([[left, right]], device=device), th


training_steps = 0


def trainQNetwork(episode_nb, learning_rate, batch_size=BATCH_SIZE, load_weights=None, save_name='tests', beatmap_name=None, star=1, frequency=10.0, eval=False):
    global training_steps
    print('Discretized x: ' + str(X_DISCRETE))
    print('Discretized y: ' + str(Y_DISCRETE))
    print('Action dim: ' + str(X_DISCRETE*Y_DISCRETE*4))
    print('X_MAX = ' + str(145 + (X_DISCRETE - 1)*DISCRETE_FACTOR))
    print('YMAX = ' + str(54 + 26 + (Y_DISCRETE - 1)*DISCRETE_FACTOR))

    # Osu routine
    process, wndw = utils.osu_routines.start_osu()
    utils.osu_routines.move_to_songs(star=star)
    if beatmap_name is not None:
        utils.osu_routines.select_beatmap(beatmap_name)
    utils.osu_routines.enable_nofail()

    q_trainer = QTrainer(batch_size=batch_size, lr=learning_rate, discrete_height=Y_DISCRETE, discrete_width=X_DISCRETE, load_weights=load_weights)

    episodes_reward = 0.0
    episode_average_reward = 0.0
    c = 0
    k = 0
    for i in range(episode_nb):
        utils.osu_routines.launch_random_beatmap()
        time.sleep(0.5)
        previous_score = torch.tensor(0.0, device=device)
        previous_acc = torch.tensor(100.0, device=device)
        controls_state = torch.tensor([[0.5, 0.5, 0.0, 0.0]], device=device)
        state = utils.screen.get_game_screen(q_trainer.screen).unsqueeze_(0).sum(1, keepdim=True) / 3.0

        thread = None
        th = None
        start = time.time()
        for step in range(MAX_STEPS):
            k += 1
            step_time_prev = time.time()
            with torch.no_grad():
                if eval:  # Choose greedy policy if tests
                    action = q_trainer.select_action(state, controls_state)
                else:  # Else choose an epsilon greedy policy with decaying epsilon
                    sample = random()
                    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * training_steps / EPS_DECAY)
                    training_steps += 1
                    if sample > eps_threshold:
                        action = q_trainer.select_action(state, controls_state)
                    else:
                        x = q_trainer.noise()  # Normal distribution mean=0.5, clipped in [0, 1[
                        action = torch.tensor([int(x[0] * X_DISCRETE) + X_DISCRETE * int(x[1]*Y_DISCRETE) + X_DISCRETE*Y_DISCRETE*int(x[2]*4)], device=device)

                new_controls_state, th = perform_discrete_action(action, q_trainer.hc, th)
                new_state = utils.screen.get_game_screen(q_trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0
                score, acc = utils.OCR.get_score_acc(q_trainer.screen, q_trainer.score_ocr, q_trainer.acc_ocr, wndw)

                # ToDo: Maybe unit test these two lines.
                new_x, new_y = pyautogui.position()
                new_controls_state = torch.cat((torch.tensor([[new_x/WIDTH, new_y/HEIGHT]], device=device), new_controls_state), 1)

                if (step < 15 and score == -1) or (score - previous_score > 5*(previous_score+100)):
                    score = previous_score
                if step < 15 and acc == -1:
                    acc = previous_acc
                reward = get_reward(score, previous_score, acc, previous_acc, step)
                done = (score == -1)
                if done:
                    new_state = None
                else:
                    #t = torch.squeeze(trainer.critic(state, controls_state, action))
                    #print(t)
                    th = Thread(target=q_trainer.memory.push, args=(state, action, reward, new_state, controls_state, new_controls_state))
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
            if k % 1000 == 0:
                print('Reward average over last 1000 steps: ')
                tmp = episode_average_reward/1000
                print(tmp)
                episodes_reward += tmp
                c += 1
                episode_average_reward = 0.0

            if done:
                break

            step_time_curr = time.time()
            if step_time_curr - step_time_prev < 1/frequency:
                dt = 1/frequency - (step_time_curr - step_time_prev)
                time.sleep(dt/1.1)

        end = time.time()
        delta_t = end - start
        print(str(step) + ' time steps in ' + str(delta_t) + ' s.')
        print(str(step / delta_t) + ' time_steps per second.')

        gc.collect()

        if i % TARGET_UPDATE == 0:
            q_trainer.target_q_network.load_state_dict(q_trainer.q_network.state_dict())

        if i % 20 == 0 and i > 0:
            print('Mean reward over last 20 episodes: ')
            print(episodes_reward / c)

            c = 0
            episodes_reward = 0.0
            if beatmap_name is not None:
                tmp = beatmap_name + save_name
            else:
                tmp = save_name
            q_trainer.save_model(tmp, num=i)

        # trainer.noise.reset()

        utils.osu_routines.return_to_beatmap()
        pyautogui.mouseUp(button='right')
        pyautogui.mouseUp(button='left')

    if (episode_nb - 1) % 10 != 0:
        print('Mean reward over last episodes: ')
        print(episodes_reward / c)
        if beatmap_name is not None:
            tmp = beatmap_name + save_name
        else:
            tmp = save_name
        q_trainer.save_model(tmp, num=episode_nb - 1)

    q_trainer.screen.stop()
    utils.osu_routines.stop_osu(process)


if __name__ == '__main__':
    #weights_path = ('./weights/actorbongo_09-12-2020-200.pt', './weights/criticbongo_09-12-2020-200.pt')
    weights_path = './weights/q_net_bongo_09-12-2020-20.pt'
    save_name = '_09-12-2020-'

    #trainDDPG(100, LEARNING_RATE, save_name=save_name, load_weights=None, beatmap_name="bongo", star=2)
    trainQNetwork(300, LEARNING_RATE, eval=False, load_weights=None, beatmap_name="fubuki guys", star=2, save_name=save_name, batch_size=BATCH_SIZE)

import torch
import time
import pyautogui
import gc
import pyclick
import torchvision
from threading import Thread

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
from trainer import Trainer

torch.cuda.empty_cache()

BATCH_SIZE = 5
LEARNING_RATE = 0.00001
GAMMA = 0.999
TAU = 0.0001
MAX_STEPS = 50000
WIDTH = 735
HEIGHT = 546

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

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
        thread = Thread(target=threaded_mouse_move, args=(min(int(x*WIDTH)+145, 830), 54+26, 0.12, human_clicker))
        thread.start()
    elif y*HEIGHT > 560:
        thread = Thread(target=threaded_mouse_move, args=(min(int(x*WIDTH)+145, 830), 560+26, 0.12, human_clicker))
        thread.start()
    elif x*WIDTH > 685:
        thread = Thread(target=threaded_mouse_move, args=(685+145, int(y*HEIGHT)+26, 0.12, human_clicker))
        thread.start()
    else:
        thread = Thread(target=threaded_mouse_move, args=(max(int(x*WIDTH)+145, 145), int(y*HEIGHT)+26, 0.12, human_clicker))
        thread.start()
    return torch.tensor([[x, y, left, right]], device=device)


def get_reward(score, previous_score, acc, previous_acc, x, y, step):
    if acc > previous_acc:
        bonus = torch.tensor(1.0, device=device)
    elif acc < previous_acc:
        bonus = torch.tensor(-0.5, device=device)
    else:
        if acc < 0.2 + 0.1 * (step//500):
            bonus = torch.tensor(-0.1, device=device)
        else:
            bonus = torch.tensor(0.0, device=device)
    return torch.log10(max((score - previous_score), torch.tensor(1.0, device=device))) + bonus  # - 0.005 * ((x - 0.5)**2 + (y - 0.5)**2)


## Training
def train(episode_nb, learning_rate, batch_size=BATCH_SIZE, load_weights=None, save_name='tests'):
    # Osu routine
    process, wndw = utils.osu_routines.start_osu()
    utils.osu_routines.move_to_songs(star=1)
    utils.osu_routines.enable_nofail()
    episode_average_reward = 0.0
    trainer = Trainer(load_weights=load_weights, lr=learning_rate, batch_size=batch_size, tau=TAU, gamma=GAMMA)
    k = 0
    episodes_reward = 0.0
    c = 0
    for i in range(episode_nb):
        utils.osu_routines.launch_random_beatmap()
        previous_screen = utils.screen.get_game_screen(trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0
        previous_score = torch.tensor(0.0, device=device)
        previous_acc = torch.tensor(100.0, device=device)
        controls_state = torch.tensor([[0.5, 0.5, 0.0, 0.0]], device=device)
        current_screen = utils.screen.get_game_screen(trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0
        state = current_screen - previous_screen
        start = time.time()
        thread = None
        #logger = open('./benchmark/log.txt', 'w+')
        for step in range(MAX_STEPS):
            k += 1
            action = trainer.select_exploration_action(state, controls_state)
            #action = trainer.select_exploitation_action(state, controls_state)
            previous_screen = current_screen
            new_controls_state = perform_action(action, trainer.hc)
            current_screen = (utils.screen.get_game_screen(trainer.screen).unsqueeze_(0).sum(1, keepdim=True)/3.0)
            score, acc = utils.OCR.get_score_acc(trainer.screen, trainer.score_ocr, trainer.acc_ocr, wndw)
            if (step < 15 and score == -1) or (score - previous_score > 5*(previous_score+100)):
                score = previous_score
            if step < 15 and acc == -1:
                acc = previous_acc
            reward = get_reward(score, previous_score, acc, previous_acc, controls_state[0][0], controls_state[0][1], step)
            #print(reward)
            done = (score == -1)
            if done:
                new_state = None
            else:
                new_state = current_screen - previous_screen
                #with torch.no_grad():
                #    t = torch.squeeze(trainer.critic(state, controls_state, action))
                #    print(t)
                #    torchvision.transforms.ToPILImage()(torch.squeeze(state)).save('./benchmark/' + str(step) + '__' '''+ str(t.item())''' + '_.png')
                #    logger.write(str(step) + '___' + str(t.item()) + '___' + str(controls_state) + '___' + str(action) + '\n')
                th = Thread(target=trainer.memory.push, args=(state, action, reward, new_state, controls_state, new_controls_state))
                th.start()
                # memory.push(torch.squeeze(state, 0), torch.squeeze(action, 0), torch.tensor(reward).to(device), torch.squeeze(new_state, 0))

            if thread is not None:
                thread.join()
            thread = Thread(target=trainer.optimize)
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

        end = time.time()
        delta_t = end - start
        #logger.close()
        print(str(step) + ' time steps in ' + str(delta_t) + ' s.')
        print(str(step/delta_t) + ' time_steps per second.')
        gc.collect()  # Garbage collector at each episode

        if i % 30 == 0 and i > 0:
            print('Mean reward over last 30 episodes: ')
            print(episodes_reward/c)
            c = 0
            episodes_reward = 0.0
            trainer.save_model(save_name, num=i)

        utils.osu_routines.return_to_beatmap()
        trainer.noise.reset()
        pyautogui.mouseUp(button='right')
        pyautogui.mouseUp(button='left')

    if (episode_nb - 1) % 15 != 0:
        print('Mean reward over last episodes: ')
        print(episodes_reward/c)
        trainer.save_model(save_name, num=episode_nb-1)

    trainer.screen.stop()
    utils.osu_routines.stop_osu(process)


if __name__ == '__main__':
    weights_path = ('./weights/actornew_best1_04-12-2020-14.pt', './weights/criticnew_best1_04-12-2020-14.pt')
    save_name = 'new_best1_04-12-2020-'
    train(15, LEARNING_RATE, save_name=save_name, load_weights=weights_path)


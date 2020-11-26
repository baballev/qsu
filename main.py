import torch
import time
import pyautogui
import gc
from threading import Thread

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
from trainer import Trainer

BATCH_SIZE = 5
LEARNING_RATE = 0.001
GAMMA = 0.999
TAU = 0.0001
MAX_STEPS = 5000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


## Functions
def threaded_mouse_move(x, y, t, human_clicker):
    human_clicker.move((x, y), t)
    return


def perform_action(action, human_clicker):
    t = torch.squeeze(action, 0)  # ToDo: Check for memory leak
    x, y = int(t[0]), int(t[1]*(600/1024))
    if t[2] < 512:
        pyautogui.mouseDown(button='left')
    else:
        pyautogui.mouseUp(button='left')
    if t[3] < 512: # ToDo: variabiliser
        pyautogui.mouseDown(button='right')
    else:
        pyautogui.mouseUp(button='right')
    if y < 75:
        thread = Thread(target=threaded_mouse_move, args=(min(x, 840), 75+26, 0.1, human_clicker))
        thread.start()
    elif y > 552:
        thread = Thread(target=threaded_mouse_move, args=(min(x, 830), 552+26, 0.1, human_clicker))
        thread.start()
    elif x > 840:
        thread = Thread(target=threaded_mouse_move, args=(840, y+26, 0.1, human_clicker))
        thread.start()
    else:
        thread = Thread(target=threaded_mouse_move, args=(max(x, 100), y+26, 0.1, human_clicker))
        thread.start()
    return x, y


def get_reward(score, previous_score, x, y):
    return 0.9 * max((score - previous_score), 0) - 0.0001 * ((x - 512)**2 + (y - 300)**2)


## Training
def train(episode_nb, learning_rate, load_weights=None, save_name='tests'):
    process = utils.osu_routines.start_osu()

    trainer = Trainer() # ToDo: Modify parameters
    # Osu routine
    utils.osu_routines.move_to_songs(star=1)
    utils.osu_routines.enable_nofail()

    for i in range(episode_nb):
        utils.osu_routines.launch_random_beatmap()
        current_screen = utils.screen.get_screen(trainer.screen)
        previous_score = 0
        state = torch.unsqueeze(current_screen, 0)
        state = torch.unsqueeze(torch.sum(state, 1)/3, 0)
        episode_average_reward = 0
        start = time.time()
        k = 0
        for step in range(MAX_STEPS):
            k += 1
            action = trainer.select_exploration_action(state)
            x, y = perform_action(action, trainer.hc)
            time.sleep(0.08)
            current_screen = utils.screen.get_screen(trainer.screen)
            score = utils.OCR.get_score(current_screen, trainer.ocr)

            if step < 15 and score == -1:
                score = 0
            reward = get_reward(score, previous_score, x, y)

            previous_score = score
            done = (score == -1)
            if done:
                new_state = None
            else:
                new_state = torch.unsqueeze(current_screen, 0)
                new_state = torch.unsqueeze(torch.sum(new_state, 1)/3, 0)
                th = Thread(target=trainer.memory.push, args=(state, action, reward, new_state))
                th.start()
                # memory.push(torch.squeeze(state, 0), torch.squeeze(action, 0), torch.tensor(reward).to(device), torch.squeeze(new_state, 0))

            state = new_state
            thread = Thread(target=trainer.optimize)
            thread.start()

            episode_average_reward += reward
            if k % 1000 == 0:
                print('Reward average over last 1000 steps: ')
                print(episode_average_reward/1000)
                episode_average_reward = 0

            if done:
                break

            # end = time.time()
            # delta_t += (end - start)
            # print(delta_t)
            # start = end
        end = time.time()
        delta_t = end - start
        print(str(step) + ' time steps in ' + str(delta_t) + ' s.')
        print(str(step/delta_t) + ' time_steps per second.')
        # print('Average episode reward: ' + str(episode_average_reward))
        # print('Average time(s) per step: ' + str(delta_t))

        gc.collect()  # Garbage collector at each episode

        if i % 10 == 0 and i > 0:
            trainer.save_model(save_name)
            print(utils.noise.t)

        utils.osu_routines.return_to_beatmap()

    if (episode_nb - 1) % 10 != 0:
        trainer.save_model(save_name)

    trainer.screen.stop()
    utils.osu_routines.stop_osu(process)


if __name__ == '__main__':
    weights_path = ('./weights/first_tests3_actor5.pt', './weights/first_tests3_critic5.pt')
    save_name = 'trash'
    train(5, LEARNING_RATE, save_name=save_name)
    print(utils.noise.t)







import torch
import os
import PIL
import time
import random
import math
import pyclick
import pyautogui
import gc
import torch.nn.functional as F
from threading import Thread
import pytessy

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
import utils.network_updates
import models
from memory import ReplayMemory

BATCH_SIZE = 5
LEARNING_RATE = 0.001
GAMMA = 0.999
TAU = 0.001
MAX_STEPS = 5000000
# MAX_STEPS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


## Functions
def select_exploration_action(state, actor):  # Check if the values are ok
    action = actor(state).detach()
    new_action = action + utils.noise.action_noise().to(device) * (actor.width / 2)
    return new_action


def threaded_mouse_move(x, y, t, human_clicker):
    human_clicker.move((x, y), t)
    return


def save_model(model, file_name, number, qual):
    torch.save(model.state_dict(), './weights/' + file_name + qual + str(number) + '.pt')
    print('Model saved to : ' + './weights/' + file_name + qual + str(number) + '.pt')
    return


def load_models(weights_path, actor, critic, target_actor, target_critic):
    actor.load_state_dict(torch.load(weights_path[0]))
    critic.load_state_dict(torch.load(weights_path[1]))
    utils.network_updates.hard_copy(target_actor, actor)
    utils.network_updates.hard_copy(target_critic, critic)
    return


def perform_action(action, human_clicker, actor):
    t = torch.squeeze(action, 0)  # ToDo: Check for memory leak
    x, y = int(t[0]), int(t[1]*(600/1024))
    if t[2] < (actor.width / 2):
        pyautogui.mouseDown(button='left')
    else:
        pyautogui.mouseUp(button='left')
    if t[3] < (actor.width / 2):
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


def optimize(actor_optimizer, critic_optimizer, memory, actor, critic, target_actor, target_critic):
    if len(memory) < BATCH_SIZE:
        return
    s1, a1, r1, s2 = memory.sample(BATCH_SIZE)

    # ---------- Critic ----------
    a2 = target_actor(s2).detach()
    next_val = torch.squeeze(target_critic(s2, a2).detach())
    y_expected = r1 + GAMMA * next_val  # y_exp = r + gamma * Q'(s2, pi'(s2))
    y_predicted = torch.squeeze(critic(s1, a1))  # y_exp = Q(s1, a1)
    loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
    # print('los_critic')
    # print(loss_critic)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()

    # ---------- Actor ----------
    pred_a1 = actor(s1)
    loss_actor = -torch.sum(critic(s1, pred_a1))
    # print('loss_actor')
    # print(loss_actor)
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()

    utils.network_updates.soft_update(target_actor, actor, TAU)
    utils.network_updates.soft_update(target_critic, critic, TAU)
    # Todo: Add verbose ?


## Training
def train(episode_nb, learning_rate, load_weights=None, save_name='tests'):
    process = utils.osu_routines.start_osu()

    # Networks & optimizers

    actor = models.Actor().to(device)
    target_actor = models.Actor().to(device)
    utils.network_updates.hard_copy(target_actor, actor)
    critic = models.Critic().to(device)
    target_critic = models.Critic().to(device)
    utils.network_updates.hard_copy(target_critic, critic)
    '''
    actor = models.Trash1().to(device)
    target_actor = models.Trash1().to(device)
    utils.network_updates.hard_copy(target_actor, actor)
    critic = models.Trash2().to(device)
    target_critic = models.Trash2().to(device)
    utils.network_updates.hard_copy(target_critic, critic)
    '''
    if load_weights is not None:
        load_models(load_weights, actor, critic, target_actor, target_critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), learning_rate)

    memory = ReplayMemory(1000)

    # Osu routine
    utils.osu_routines.move_to_songs(star=1)
    # utils.osu_routines.reset_mods()
    utils.osu_routines.enable_nofail()

    screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
    ocr = utils.OCR.init_OCR()
    hc = pyclick.HumanClicker()
    k = 0
    for i in range(episode_nb):
        utils.osu_routines.launch_random_beatmap()
        time.sleep(1)
        current_screen = utils.screen.get_screen(screen)
        previous_score = 0
        state = torch.unsqueeze(current_screen, 0)
        state = torch.unsqueeze(torch.sum(state, 1)/3, 0)
        episode_average_reward = 0
        start = time.time()
        for step in range(MAX_STEPS):
            k += 1
            action = select_exploration_action(state, actor)

            x, y = perform_action(action, hc, actor)
            current_screen = utils.screen.get_screen(screen)
            score = utils.OCR.get_score(current_screen, ocr)

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
                th = Thread(target=memory.push, args=(state, action, reward, new_state))
                th.start()
                # memory.push(torch.squeeze(state, 0), torch.squeeze(action, 0), torch.tensor(reward).to(device), torch.squeeze(new_state, 0))

            state = new_state
            thread = Thread(target=optimize, args=(actor_optimizer, critic_optimizer, memory, actor, critic, target_actor, target_critic))
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
        print('Average episode reward: ' + str(episode_average_reward))
        # print('Average time(s) per step: ' + str(delta_t))

        gc.collect()  # Garbage collector at each episode

        if i % 10 == 0 and i > 0:
            save_model(target_actor, save_name, i, '_actor')
            save_model(target_critic, save_name, i, '_critic')
            print(utils.noise.t)

        utils.osu_routines.return_to_beatmap()

    if (episode_nb - 1) % 10 != 0:
        save_model(target_actor, save_name, episode_nb, '_actor')
        save_model(target_critic, save_name, episode_nb, '_critic')

    screen.stop()
    utils.osu_routines.stop_osu(process)


if __name__ == '__main__':
    weights_path = ('./weights/first_tests3_actor5.pt', './weights/first_tests3_critic5.pt')
    save_name = 'first_tests_23-11'
    train(1, LEARNING_RATE)
    print(utils.noise.t)







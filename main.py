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

import utils.screen
import utils.osu_routines
import utils.OCR
import utils.noise
import utils.network_updates
import models
from memory import ReplayMemory

BATCH_SIZE = 5
LEARNING_RATE = 0.01
GAMMA = 0.999
TAU = 0.01
MAX_STEPS = 5000000
# MAX_STEPS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## IMPORTANT NOTE: THE LARGE MAJORITY of the code was taken or inspired from:
## https://github.com/vy007vikas/PyTorch-ActorCriticRL/
## All credits go to vy007vikas for the nice Pytorch continuous action actor-critic DDPG she/he/they made.


## Functions
def select_exploration_action(state, actor):  # Check if the values are ok
    action = actor(state).detach()
    new_action = action.data + utils.noise.action_noise().to(device) * (actor.width / 2)
    return new_action


mouse_state = [False, False]  # 0 up, 1 down


def change_click(side, state):  # left or right
    if side == 'left':
        if state[0]:
            pyautogui.mouseUp(button='left')
            state[0] = False
        else:
            pyautogui.mouseDown(button='left')
            state[0] = True
    else:
        if state[1]:
            pyautogui.mouseUp(button='right')
            state[1] = False
        else:
            pyautogui.mouseDown(button='right')
            state[1] = True


def threaded_mouse_move(x, y, t, human_clicker):
    human_clicker.move((x, y), t)
    return


def save_model(model, file_name, number, qual):
    torch.save(model.state_dict(), './weights/' + file_name + qual + str(number) + '.pt')
    print('Model saved to : ' + './weights/' + file_name + qual + str(number) + '.pt')
    return


def load_models(weights_path, actor, critic, target_actor, target_critic):
    actor.load_state_dict(torch.load(weights_path[0]))
    critic.load_state_dict(torch.laod(weights_path[1]))
    utils.network_updates.hard_copy(target_actor, actor)
    utils.network_updates.hard_copy(target_critic, critic)
    return


def perform_action(action, human_clicker, actor):
    t = torch.squeeze(action, 0).cpu()  # ToDo: Check for memory leak
    # print(t)
    x, y = int(t[0].item()), int(t[1].item()*(1024/600))
    # print('(x, y) = (' + str(x) + ', ' + str(y) + ')')
    changed = False
    if t[2].item() < (actor.width / 2):
        change_click('left', mouse_state)
        changed = True
    if t[3].item() < (actor.width / 2):
        change_click('right', mouse_state)
        changed = True
    del t
    if y < 50 or y > 600:
        return x, y, changed
    elif x > 1024:
        return x, y, changed
    else:
        thread = Thread(target=threaded_mouse_move, args=(x, y+29, 0.1, human_clicker))
        thread.start()
        return x, y, changed


def get_reward(score, previous_score, x, y, previous_x, previous_y, changed):
    ALPHA = 0.00001
    BETA = 0.00001
    C = 0.0001
    if changed:
        value = 1
    else:
        value = 0
    return max((score - previous_score), 0) - ALPHA * (((previous_x - x) ** 2 + (previous_y - y) ** 2)) - BETA * (
                max(x - 1024, 0) + max(y - 600, 0)) + C * value


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
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()

    # ---------- Actor ----------
    pred_a1 = actor(s1)
    loss_actor = -torch.sum(critic(s1, pred_a1))
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()

    utils.network_updates.soft_update(target_actor, actor, TAU)
    utils.network_updates.soft_update(target_critic, critic, TAU)
    # Todo: Add verbose ?


## Training
def train(episode_nb, learning_rate, load_weights=None):
    process = utils.osu_routines.start_osu()

    # Networks & optimizers
    actor = models.Actor().to(device)
    target_actor = models.Actor().to(device)
    utils.network_updates.hard_copy(target_actor, actor)
    critic = models.Critic().to(device)
    target_critic = models.Critic().to(device)
    utils.network_updates.hard_copy(target_critic, critic)

    if load_weights is not None:
        load_models(load_weights, actor, critic, target_actor, target_critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), learning_rate)

    memory = ReplayMemory(400)

    # Osu routine
    utils.osu_routines.move_to_songs(star=1)
    # utils.osu_routines.reset_mods()
    utils.osu_routines.enable_nofail()

    screen = utils.screen.init_screen(capture_output="pytorch_float_gpu")
    hc = pyclick.HumanClicker()
    for i in range(episode_nb + 1):
        utils.osu_routines.launch_random_beatmap()
        time.sleep(1)
        last_screen = utils.screen.get_screen(screen)
        current_screen = utils.screen.get_screen(screen)
        previous_score = 0
        state = torch.unsqueeze((current_screen - last_screen).permute(2, 0, 1), 0)

        fail_read_counter = 0
        episode_average_reward = 0
        for step in range(MAX_STEPS):
            action = select_exploration_action(state, actor)
            previous_x, previous_y = pyautogui.position()
            previous_y -= 29
            x, y, changed = perform_action(action, hc, actor)
            score = utils.OCR.get_score(screen)
            if step < 50 and score == -1:
                score = 0
            reward = get_reward(score, previous_score, x, y, previous_x, previous_y, changed)
            if reward > 1000:
                reward = 0
            last_screen = current_screen
            current_screen = utils.screen.get_screen(screen)
            # current_screen = screen.get_latest_frame() ToDo: test this one, does it gets the last screenshot that has been done and is it in the buffer?
            #  ToDo: it could prevent small differences between the image used for score and the new state

            if score == -1:
                if fail_read_counter > 8:
                    done = True  # ToDo: Code an OCR function to check?
                else:
                    fail_read_counter += 1
                    continue
            else:
                done = False
            if done:
                new_state = None
            else:
                new_state = torch.unsqueeze((current_screen - last_screen).permute(2, 0, 1), 0)
                memory.push(torch.squeeze(state, 0), torch.squeeze(action, 0), torch.tensor(reward).to(device),
                            torch.squeeze(new_state, 0))

            if step % 10 == 0:
                optimize(actor_optimizer, critic_optimizer, memory, actor, critic, target_actor, target_critic)

            episode_average_reward += reward

            previous_score = score
            if done:
                episode_average_reward = episode_average_reward/step
                break

        print('Average episode reward: ' + str(episode_average_reward))

        gc.collect()  # Garbage collector at each episode
        # ToDo: Save / Load model
        # ToDo: Verbose
        if i%5 == 0:
            save_model(target_actor, 'first_tests', i, '_actor')
            save_model(target_critic, 'first_tests', i, '_critic')

        utils.osu_routines.return_to_beatmap()

    if episode_nb % 5 != 0:
        save_model(target_actor, 'first_tests', episode_nb, '_actor')
        save_model(target_critic, 'first_tests', episode_nb, '_critic')


    screen.stop()
    utils.osu_routines.stop_osu(process)


if __name__ == '__main__':
    train(12, LEARNING_RATE)
    print(utils.noise.t)
    # process = utils.osu_routines.start_osu()
    # import utils.mouse_locator
    # utils.mouse_locator.mouse_locator()
    # utils.osu_routines.stop_osu(process)

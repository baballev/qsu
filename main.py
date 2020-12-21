import math
from random import random, randint
import torch
import time
import pyautogui
import gc
from threading import Thread

import environment
from trainer import QTrainer

torch.set_printoptions(sci_mode=False)

pyautogui.MINIMUM_DURATION = 0.0
pyautogui.MINIMUM_SLEEP = 0.0
pyautogui.PAUSE = 0.0

BATCH_SIZE = 20
LEARNING_RATE = 0.0001
GAMMA = 0.999
MAX_STEPS = 25000
WIDTH = 878
HEIGHT = 600
STACK_SIZE = 4

EPS_START = 0.99
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 2000  # Number of steps

DISCRETE_FACTOR = 10
X_DISCRETE = 685 // DISCRETE_FACTOR + 1
Y_DISCRETE = (560 - 54) // DISCRETE_FACTOR + 1

PIXEL_SKIP = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    env = environment.OsuEnv(X_DISCRETE, Y_DISCRETE, DISCRETE_FACTOR, WIDTH, HEIGHT, STACK_SIZE, star=star, beatmap_name=beatmap_name, no_fail=False, skip_pixels=PIXEL_SKIP)
    q_trainer = QTrainer(env, batch_size=batch_size, lr=learning_rate, load_weights=load_weights)

    episode_average_reward = 0.0
    k = 0
    reward = torch.tensor(0.0, device=device)
    for i in range(episode_nb):
        controls_state, state = env.reset(reward)
        env.launch_episode(reward)

        episode_reward = 0.0
        thread = None
        start = time.time()
        for steps in range(MAX_STEPS):
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
                        x = q_trainer.noise()  # Normal distribution mean=0.5, clipped in [0, 1[ & uniform distrib
                        action = torch.tensor([int(x[0] * X_DISCRETE) + X_DISCRETE * int(
                            x[1] * Y_DISCRETE) + X_DISCRETE * Y_DISCRETE * int(x[2] * 4)], device=device)
                new_state, new_controls_state, reward, done = env.step(action, steps)
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

            state = new_state
            controls_state = new_controls_state
            episode_reward += reward

            if done:
                break

        end = time.time()
        delta_t = end - start
        print(str(steps) + ' time steps in ' + str(delta_t) + ' s.')
        print(str(steps / delta_t) + ' time_steps per second.')

        gc.collect()

        episode_average_reward += episode_reward
        q_trainer.avg_reward_plotter.step(episode_reward)
        if i > 0 and i % 50 == 0:
            q_trainer.avg_reward_plotter.fit()
        q_trainer.avg_reward_plotter.show()

        if i % 250 == 0:
            q_trainer.plotter.fig.savefig('average_loss' + str(i) + '.png')
            q_trainer.avg_reward_plotter.fig.savefig('episode_reward' + str(i) + '.png')

        if k % TARGET_UPDATE == 0:
            q_trainer.target_q_network.load_state_dict(q_trainer.q_network.state_dict())
            print(k)

        if i % 100 == 0 and i > 0:
            print('Mean reward over last 100 episodes: ')
            print(episode_average_reward / 100)

            episode_average_reward = 0.0
            if beatmap_name is not None:
                tmp = beatmap_name + save_name
            else:
                tmp = save_name
            q_trainer.save_model(tmp, num=i)

    if (episode_nb - 1) % 5 != 0:
        if beatmap_name is not None:
            tmp = beatmap_name + save_name
        else:
            tmp = save_name
        q_trainer.save_model(tmp, num=episode_nb - 1)
        q_trainer.plotter.fig.savefig('average_loss' + str(episode_nb-1) + '.png')
        q_trainer.avg_reward_plotter.fig.savefig('average_reward' + str(episode_nb-1) + '.png')

    q_trainer.memory.save('./memory.pck')
    env.stop()


if __name__ == '__main__':
    weights_path = './weights/q_net_fubuki guysReboot_12-12-2020-210.pt'
    save_name = '_20-12-2020-'
    trainQNetwork(5000, LEARNING_RATE, evaluation=False, load_weights=None, beatmap_name="golden time tv", star=2,
                  save_name=save_name, batch_size=BATCH_SIZE)

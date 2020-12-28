import math
from random import random, randint
import torch
import time
import pyautogui
import gc
import gym
from threading import Thread

import environment
from trainer import QTrainer, RainbowTrainer

torch.set_printoptions(sci_mode=False)

pyautogui.MINIMUM_DURATION = 0.0
pyautogui.MINIMUM_SLEEP = 0.0
pyautogui.PAUSE = 0.0

BATCH_SIZE = 32
LEARNING_RATE = 0.00005  # Double DQN: 0.00025, Prioritized Replay -> Double DQN / 4
GAMMA = 0.999
MAX_STEPS = 25000
WIDTH = 878
HEIGHT = 600
STACK_SIZE = 4

DISCRETE_FACTOR = 10
X_DISCRETE = 685 // DISCRETE_FACTOR + 1
Y_DISCRETE = (560 - 54) // DISCRETE_FACTOR + 1

PIXEL_SKIP = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_infos():
    print('Discretized x: ' + str(X_DISCRETE))
    print('Discretized y: ' + str(Y_DISCRETE))
    print('Action dim: ' + str(X_DISCRETE * Y_DISCRETE * 4))
    print('X_MAX = ' + str(145 + (X_DISCRETE - 1) * DISCRETE_FACTOR))
    print('YMAX = ' + str(54 + 26 + (Y_DISCRETE - 1) * DISCRETE_FACTOR))


def log_episodes(i, q_trainer, steps, delta_t, k):  # TODO : Make a logging utils and use log function
    if i <= 1 or i % 30 == 0:
        print(str(steps) + ' steps in ' + str(delta_t) + ' s.' + '  -->   %.4f' % (steps / delta_t) + ' steps / s')

    if i > 0 and i % 50 == 0:
        q_trainer.avg_reward_plotter.fit()
    q_trainer.avg_reward_plotter.show()

    if i > 0 and i % 250 == 0:
        q_trainer.plotter.fig.savefig('average_loss' + str(i) + '.png')
        q_trainer.avg_reward_plotter.fig.savefig('episode_reward' + str(i) + '.png')

    if i % 30 == 0:
        print("Episode %d" % (i+1))
        print("Time step %d" % k)
        print("Schedule: %f" % q_trainer.scheduler.value(k))


def busy_wait(freq, previous_t):
    while time.time() - previous_t < 1/(freq+0.5):
        time.sleep(0.001)


def trainQNetwork(episode_nb, learning_rate, batch_size=BATCH_SIZE, load_weights=None, save_name='tests',
                  beatmap_name=None, star=1, evaluation=False, human_off_policy=False, load_memory=None, no_fail=False,
                  initial_p=1.0, end_p=0.05, decay_p=2000000, target_update=30000, init_k=0, min_experience=25000):
    print_infos()
    if evaluation:
        learning_rate = 0.0

    env = environment.OsuEnv(X_DISCRETE, Y_DISCRETE, DISCRETE_FACTOR, WIDTH, HEIGHT, STACK_SIZE, star=star,
                             beatmap_name=beatmap_name, no_fail=no_fail, skip_pixels=PIXEL_SKIP)
    q_trainer = QTrainer(env, batch_size=batch_size, lr=learning_rate, gamma=GAMMA, initial_p=initial_p, end_p=end_p,
                         decay_p=decay_p, load_weights=load_weights, load_memory=load_memory,
                         min_experience=min_experience, gradient_clipping_norm=10.0)

    episode_average_reward = 0.0
    k = init_k
    reward = torch.tensor(0.0, device=device)
    for i in range(episode_nb):
        controls_state, state = env.reset(reward)
        env.launch_episode(reward)

        episode_reward = 0.0
        thread = None
        start = time.time()
        for steps in range(MAX_STEPS):
            previous_t = time.time()
            k += 1
            with torch.no_grad():
                if not human_off_policy:
                    if evaluation:  # Choose greedy policy if tests
                        action = q_trainer.select_action(state, controls_state)
                    else:  # Else choose an epsilon greedy policy with decaying epsilon
                        if k > min_experience:
                            sample = random()
                            if sample > q_trainer.scheduler.value(k):
                                action = q_trainer.select_action(state, controls_state)
                            else:
                                action = q_trainer.random_action(X_DISCRETE, Y_DISCRETE)
                        else:
                            action = q_trainer.random_action(X_DISCRETE, Y_DISCRETE)

                    new_state, new_controls_state, reward, done = env.step(action, steps)
                else:
                    action, new_state, new_controls_state, reward, done = env.observe(steps)

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

            busy_wait(freq=12.0, previous_t=previous_t)
            if done:
                break
        end = time.time()
        delta_t = end - start

        gc.collect()

        episode_average_reward += episode_reward
        q_trainer.avg_reward_plotter.step(episode_reward)
        log_episodes(i, q_trainer, steps, delta_t, k)

        if k % target_update == 0:
            q_trainer.target_q_network.load_state_dict(q_trainer.q_network.state_dict())

        if i % 30 == 0 and i > 0:
            print('Mean reward over last 30 episodes: ')
            print(episode_average_reward / 30)

            episode_average_reward = 0.0
            if beatmap_name is not None:
                tmp = beatmap_name + save_name
            else:
                tmp = save_name
            q_trainer.save_model(tmp, num=i)

    if (episode_nb - 1) % 15 != 0:
        if beatmap_name is not None:
            tmp = beatmap_name + save_name
        else:
            tmp = save_name
        q_trainer.save_model(tmp, num=episode_nb - 1)
        q_trainer.plotter.fig.savefig('average_loss' + str(episode_nb-1) + '.png')
        q_trainer.avg_reward_plotter.fig.savefig('average_reward' + str(episode_nb-1) + '.png')
    print('k: %d' % k)
    env.stop()


def RainbowManiaTrain(lr=0.00005, batch_size=32, gamma=0.999, omega=0.5, beta=0.4, sigma=0.1, eps=1.5e-4, n=3, atoms=51,
                      max_timesteps=50000000, learn_start=100000, stack_size=4, norm_clip=10, save_freq=50000,
                      save_path='weights/Rainbow_test', target_update_freq=80000, star=4, beatmap_name=None,
                      width=1024, height=600, skip_pixels=4, num_actions=128):
    priority_weight_increase = (1 - beta) / (max_timesteps - learn_start)

    env = environment.ManiaEnv(height=height, width=width, stack_size=stack_size, star=star, beatmap_name=beatmap_name, num_actions=num_actions, skip_pixels=skip_pixels)
    trainer = RainbowTrainer(env, batch_size=batch_size, lr=lr, gamma=gamma, omega=omega, beta=beta, sigma=sigma, n=n,
                             eps=eps, atoms=atoms, norm_clip=norm_clip)

    reward = 0.0
    need_save = False
    need_update = False
    done = True
    start = time.time()
    count = 0
    for t in range(max_timesteps):
        if done:
            if t > 0:
                end = time.time()
                print((t-count)/(end - start))
                count = t
            state = env.reset(reward)
            env.launch_episode(reward)
            if need_save:
                trainer.save(save_path + str(t) + ".pt")
                # TODO: Memory saving with bz2

            if need_update:
                start = time.time()
                trainer.update_target_net()
                end = time.time()
                print(end-start)

            start = time.time()

        trainer.reset_noise()
        action = trainer.select_action(state)  # ToDo print pour débug
        next_state, reward, done = env.step(action)
        reward = max(min(reward, 1.0), -1.0)  # Reward clipping
        trainer.memory.append(state[-1], action, reward, done)

        if t >= learn_start:
            trainer.memory.priority_weight = min(trainer.memory.priority_weight + priority_weight_increase, 1)
            trainer.optimize()

        if t % target_update_freq == 0 and t > 0:
            need_update = True  # TODO: CHECK IF THIS TAKES TIME, IF NOT PUT IT DURING GAMEPLAY

        if t % save_freq == 0 and t > 0:
            need_save = True


if __name__ == '__main__':
    '''
    weights_path = './weights/q_net__21-12-2020-14.pt'
    memory_path = './memory.pck'
    save_name = '_25-12-2020-'
    trainQNetwork(50, LEARNING_RATE, evaluation=False, load_weights=None, beatmap_name="sink", star=7,
                  save_name=save_name, batch_size=BATCH_SIZE, human_off_policy=False, no_fail=True,
                  initial_p=1.0, end_p=0.05, decay_p=4000000, target_update=30000, init_k=0, min_experience=50)
    '''
    RainbowManiaTrain(star=4, beatmap_name="todestrieb", num_actions=2**4)

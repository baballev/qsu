import pickle
from random import random, randint
import torch
import time
import pyautogui
import gc
import numpy as np
from threading import Thread

from tqdm import tqdm

import environment
from trainer import QTrainer, RainbowTrainer, TaikoTrainer

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

            #busy_wait(freq=12.0, previous_t=previous_t)
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


def RainbowManiaTrain(lr=0.0000625, batch_size=32, gamma=0.999, omega=0.5, beta=0.4, sigma=0.1, eps=1.5e-4, n=3, atoms=51,
                      max_timestep=int(5e7), learn_start=80000, stack_size=4, norm_clip=10.0, save_freq=50000,
                      model_save_path='weights/Rainbow_test', memory_save_path='weights/memory.zip', target_update_freq=80000,
                      star=4, beatmap_name=None, width=380, height=600, skip_pixels=4, num_actions=128, no_fail=False,
                      load_weights=None, load_memory=None, Vmin=-10, Vmax=10, resume_start=0, load_stats=None,
                      save_stats='./stats.pkl', learning_freq=1, load_optimizer=None, optimizer_path='weights/opti.pt',
                      evaluation=False, data_efficient=False):

    priority_weight_increase = (1 - beta) / (max_timestep - learn_start - resume_start)

    env = environment.ManiaEnv(height=height, width=width, stack_size=stack_size, star=star, beatmap_name=beatmap_name,
                               num_actions=num_actions, skip_pixels=skip_pixels, no_fail=no_fail)
    trainer = RainbowTrainer(env, batch_size=batch_size, lr=lr, gamma=gamma, omega=omega, beta=beta, sigma=sigma, n=n,
                             eps=eps, atoms=atoms, norm_clip=norm_clip, load_weights=load_weights, load_memory=load_memory,
                             Vmin=Vmin, Vmax=Vmax, load_optimizer=load_optimizer, data_efficient=data_efficient)
    if load_stats is None:
        stat = {'episode_reward': []}
    else:
        with open(load_stats, 'rb') as f:
            stat = pickle.load(f)

    reward = 0.0
    need_save = False
    done = True
    count = 0
    thread = None

    for t in tqdm(range(resume_start, max_timestep), desc="Timestep", unit='step', unit_scale=True):
        if done:
            if t > resume_start:
                count += 1
                trainer.avg_reward_plotter.step(episode_reward)
                trainer.avg_reward_plotter.show()
                trainer.plotter.show()
                stat['episode_reward'].append(episode_reward.item())
                if count % 100 == 0:
                    print('   -  Mean reward over last 100 episodes: %.4f' % np.array(stat['episode_reward'][max(-100, -len(stat['episode_reward'])):]).mean())
            if need_save:
                trainer.save(model_save_path + str(t) + ".pt", memory_save_path, optimizer_path)
                with open(save_stats, 'wb') as f:
                    pickle.dump(stat, f, protocol=4)
                need_save = False

            episode_reward = 0.0
            gc.collect()
            state = env.reset(reward)
            env.launch_episode(reward)
        trainer.reset_noise()
        action = trainer.select_action(state)
        next_state, reward, done = env.step(action)
        reward = max(min(reward, 1.0), -1.0)  # Reward clipping
        episode_reward += reward
        if env.episode_counter > 0:  # Skip first episode because of latency issues
            trainer.memory.append(state[-1], action, reward, done)

        if t >= learn_start and t % learning_freq == 0 and not evaluation:
            trainer.memory.priority_weight = min(trainer.memory.priority_weight + priority_weight_increase, 1)
            if thread is not None:
                thread.join()
            thread = Thread(target=trainer.optimize)
            thread.start()

        if t % target_update_freq == 0 and t > 0:
            trainer.update_target_net()

        if t % save_freq == 0 and t > 0:
            need_save = True


def TaikoTrain(lr=0.00005, batch_size=32, stack_size=1, skip_pixels=4, save_freq=15000, episode_nb=5,
                target_update_freq=5000, star=None, beatmap_name=None, min_experience=1200, root_dir='./weights',
                evaluation=False):

    env = environment.TaikoEnv(stack_size=stack_size, star=star, beatmap_name=beatmap_name, skip_pixels=skip_pixels)
    tt = TaikoTrainer(env, batch_size=batch_size, lr=lr, gamma=GAMMA, root_dir=root_dir, min_experience=min_experience, norm_clip=10.0)

    need_update = False
    need_save = False
    for episode in range(episode_nb):
        state = env.reset()
        env.launch_episode()
        episode_reward = 0.0
        for steps in range(MAX_STEPS):
            if evaluation:
                action = tt.select_action(state)
            else:
                action = tt.select_explo_action(state)

            new_state, reward, done = env.step(action)
            episode_reward += reward

            if not done:
                tt.memory.push(state, action, reward, new_state)
            else:
                break

            tt.optimize()

            state = new_state

            # These boolean allow the program to wait for the end of the episode before performing the updates or the save to avoid latency while the agent is playing
            if tt.steps_done % target_update_freq == 0:
                need_update = True
            if tt.steps_done % save_freq == 0:
                need_save = True

        if need_update:
            tt.update_target()
            need_update = False
        if need_save:
            tt.save()
            need_save = False
        print(steps)
        tt.avg_reward_plotter.step(episode_reward)
        tt.avg_reward_plotter.show()

    tt.stop()


if __name__ == '__main__':
    '''
    weights_path = './weights/q_net__21-12-2020-14.pt'
    memory_path = './memory.pck'
    save_name = '_25-12-2020-'
    trainQNetwork(50, LEARNING_RATE, evaluation=False, load_weights=None, beatmap_name="sink", star=7,
                  save_name=save_name, batch_size=BATCH_SIZE, human_off_policy=False, no_fail=True,
                  initial_p=1.0, end_p=0.05, decay_p=4000000, target_update=30000, init_k=0, min_experience=50)
    '''
    '''
    RainbowManiaTrain(star=3, beatmap_name="bongo", num_actions=2**4, model_save_path="weights/Taiko_Bongo-31-05-2021_3stars",
                      learn_start=1600, load_weights=None, load_memory=None, batch_size=32, max_timestep=int(1e6),
                      memory_save_path='./weights/memory28-03-2021.zip', Vmin=-1, Vmax=10, resume_start=0, target_update_freq=5000,
                      load_stats=None, save_freq=5000, save_stats='./stats/stats-28-03-2021.pkl', learning_freq=1, lr=0.0001,
                      load_optimizer=None, optimizer_path='./weights/opti.pt', evaluation=False, n=20, data_efficient=True)
    '''

    TaikoTrain(root_dir='./weights/Taiko/', episode_nb=50, min_experience=35)
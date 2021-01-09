import pickle

import matplotlib.pyplot as plt
import time
import numpy as np
import utils.osu_routines
import win32gui
import datetime
from threading import Thread

window_idx = 1


def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


class LivePlot:
    def __init__(self, min_x=-500, max_x=0, min_y=-10, num_points=500, max_y=10, window_x=1900, window_y=90, x_axis='200 of steps', y_axis='reward'):
        global window_idx
        if window_idx == 1:
            th = Thread(target=utils.osu_routines.shut_annoying_window)
            th.start()
        self.min_x = min_x
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ylim(min_y, max_y)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        self.x = np.linspace(min_x, max_x, num_points)
        self.y = [0 for _ in range(num_points)]
        self.line, = self.ax.plot(self.x, self.y)
        self.fig.show()
        if window_idx == 1:
            th.join()
        time.sleep(0.5)
        self.window = win32gui.FindWindow(None, "Figure " + str(window_idx))
        window_idx += 1
        width, height = win32gui.GetWindowRect(self.window)[2] - win32gui.GetWindowRect(self.window)[0], \
                        win32gui.GetWindowRect(self.window)[3] - win32gui.GetWindowRect(self.window)[1]
        win32gui.MoveWindow(self.window, window_x, window_y, width, height, False)
        self.show()
        time.sleep(0.1)
        self.show()
        self.t = 0
        self.curve = None
        self.text = None
        utils.osu_routines.relocate()

    def step(self, reward):
        #print(reward)
        self.y.append(reward.item())
        self.y.pop(0)
        self.line.set_ydata(self.y)
        self.t += 1

    def show(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def fit(self):
        if self.curve is not None:
            l = self.curve.pop(0)
            l.remove()
            del l
        a, b = np.polyfit(self.x[max(-self.t, self.min_x):], self.y[max(-self.t, self.min_x):], 1)
        if self.text is not None:
            self.text.set_text('%.4f'%a + '.x + ' + '%.4f'%b)
        else:
            self.text = self.ax.text(-450, 2, '%.4f'%a + '.x + ' + '%.4f'%b)
        self.curve = self.ax.plot(self.x, a*self.x+b)


if __name__ == '__main__':
    fig = plt.figure()
    with open('../stats/stats_08-01-2021-first_try.pkl', 'rb') as f:
        stats = pickle.load(f)
    ax = plt.subplot(111)
    n = len(stats['episode_reward'])
    x = np.arange(0, n, 1)
    y = np.array(stats['episode_reward'])
    ax.plot(x, y)
    y2 = np.convolve(y, np.ones(10), 'valid')/10
    ax.plot(x[:-9], y2)
    plt.show()



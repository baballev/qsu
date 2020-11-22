import win32gui
import os
import subprocess
import psutil
import time
import pyclick
import random
import pyautogui

from utils.osu_config import OSU_FOLDER_PATH, swap_to_AI, restore_player, USERNAME, AI_NAME


def start_osu():
    try:
        for proc in psutil.process_iter():
            if proc.name() == 'osu!.exe':
                proc.kill()
        time.sleep(0.5)
        swap_to_AI(AI_NAME, USERNAME)
        process = subprocess.Popen(OSU_FOLDER_PATH + 'osu!.exe')
        print('Osu! started with pid: ' + str(process.pid))
        p = psutil.Process(process.pid)
        time.sleep(3)
        osu_window = win32gui.FindWindow(None, "osu!")
        width, height = win32gui.GetWindowRect(osu_window)[2] - win32gui.GetWindowRect(osu_window)[0], win32gui.GetWindowRect(osu_window)[3] - win32gui.GetWindowRect(osu_window)[1]
        print(width)
        print(height)
        win32gui.MoveWindow(osu_window, -3, 0, width, height, False)
    except Exception as e:
        print(e)
        p = None

    return p


def stop_osu(process):
    if process is None:
        return
    try:
        print('Killing osu process with expected pid: ' + str(process.pid))
        process.kill()
    except Exception as e:
        print(e)
    restore_player(USERNAME)
    return


def move_to_songs(star=1):
    hc = pyclick.HumanClicker()
    time.sleep(2)
    delta_t = random.random()
    hc.move((496, 325), delta_t*1.5+0.3)
    hc.click()
    time.sleep(0.2)
    delta_t = random.random()
    hc.move((660, 210), delta_t*1.5+0.3)
    hc.click()
    time.sleep(0.5)
    delta_t = random.random()
    hc.move((760, 250), delta_t*1.5+0.3)
    hc.click()
    delta_t = random.random()
    time.sleep(delta_t + 1)
    pyautogui.scroll(10)
    time.sleep(0.2)
    pyautogui.scroll(10)
    time.sleep(0.1)
    pyautogui.scroll(10)
    time.sleep(1.25)
    pyautogui.scroll(-10)
    time.sleep(0.1)
    pyautogui.scroll(-10)
    delta_t = random.random()
    time.sleep(0.5 + delta_t)
    hc.move((730, 110 + star*60), 1)
    hc.click()
    time.sleep(0.4)
    hc.move((450, 320), 0.8)
    del hc
    return


def launch_random_beatmap():
    hc = pyclick.HumanClicker()
    time.sleep(0.25)
    t = random.random()
    hc.move((338, 594), 0.25 + t)
    time.sleep(0.1)
    hc.click()
    time.sleep(3.5)
    hc.move((984, 524), 0.8)
    time.sleep(0.25)
    hc.click()
    time.sleep(0.3)
    hc.move((500, 320), 0.3)
    time.sleep(0.2)
    return


def enable_nofail():
    hc = pyclick.HumanClicker()
    time.sleep(0.25)
    hc.move((275, 590), 0.4)
    time.sleep(0.1)
    hc.click()
    time.sleep(0.3)
    hc.move((385, 195), 0.5)
    time.sleep(0.05)
    hc.click()
    time.sleep(0.2)
    hc.move((350, 520), 0.6)
    time.sleep(0.15)
    hc.click()
    time.sleep(0.2)
    hc.move((150, 520), 0.7)
    time.sleep(0.2)
    return


def reset_mods():
    hc = pyclick.HumanClicker()
    time.sleep(0.25)
    hc.move((275, 590), 0.4)
    time.sleep(0.1)
    hc.click()
    time.sleep(0.3)
    hc.move((670, 450), 0.8)
    time.sleep(0.3)
    hc.click()
    time.sleep(0.2)
    hc.move((504, 518), 0.5)
    time.sleep(0.1)
    hc.click()
    time.sleep(0.4)
    hc.move((400, 360), 0.8)
    time.sleep(0.3)
    return


def return_to_beatmap():
    time.sleep(3)
    hc = pyclick.HumanClicker()
    hc.move((480, 370), 0.2)
    time.sleep(0.05)
    hc.click()
    time.sleep(0.05)
    hc.move((50, 605), 0.6)
    time.sleep(0.15)
    hc.click()
    time.sleep(0.8)
    return


## DEBUG
if __name__ == '__main__':

    osu_process = start_osu()
    time.sleep(20)
    stop_osu(osu_process)
import win32gui
import win32api
import win32con
import os
import subprocess
import psutil
import time
import pyclick
import random
import pyautogui

from utils.osu_config import OSU_FOLDER_PATH, swap_to_AI, restore_player, USERNAME, AI_NAME
import utils.OCR

KEY_DICT = {' ': 0x20,
            '0': 0x30,
            '1': 0x31,
            '2': 0x32,
            '3': 0x33,
            '4': 0x34,
            '5': 0x35,
            '6': 0x36,
            '7': 0x37,
            '8': 0x38,
            '9': 0x39,
            'a': 0x41,
            'b': 0x42,
            'c': 0x43,
            'd': 0x44,
            'e': 0x45,
            'f': 0x46,
            'g': 0x47,
            'h': 0x48,
            'i': 0x49,
            'j': 0x4A,
            'k': 0x4B,
            'l': 0x4C,
            'm': 0x4D,
            'n': 0x4E,
            'o': 0x4F,
            'p': 0x50,
            'q': 0x51,
            'r': 0x52,
            's': 0x53,
            't': 0x54,
            'u': 0x55,
            'v': 0x56,
            'w': 0x57,
            'x': 0x58,
            'y': 0x59,
            'z': 0x5A, 'backspace': 0x08}


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
        time.sleep(5)
        osu_window = win32gui.FindWindow(None, "osu!")
        width, height = win32gui.GetWindowRect(osu_window)[2] - win32gui.GetWindowRect(osu_window)[0], \
                        win32gui.GetWindowRect(osu_window)[3] - win32gui.GetWindowRect(osu_window)[1]
        print(width)
        print(height)
        win32gui.MoveWindow(osu_window, -3, 0, width, height, False)
    except Exception as e:
        print(e)
        p = None

    return p, osu_window


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
    time.sleep(1)
    hc.move((496, 325),  0.3)
    hc.click()
    time.sleep(0.4)
    hc.click()
    time.sleep(0.4)
    hc.move((660, 210), 0.3)
    hc.click()
    time.sleep(0.4)
    hc.move((760, 250), 0.3)
    hc.click()
    time.sleep(1)
    pyautogui.scroll(10)
    time.sleep(0.4)
    pyautogui.scroll(10)
    time.sleep(0.4)
    pyautogui.scroll(10)
    time.sleep(1.25)
    pyautogui.scroll(-10)
    time.sleep(0.4)
    pyautogui.scroll(-10)
    time.sleep(0.5)
    hc.move((730, 110 + star * 60), 1)
    time.sleep(0.5)
    hc.click()
    time.sleep(0.4)
    hc.move((450, 320), 0.8)
    del hc
    return


def launch_random_beatmap():
    hc = pyclick.HumanClicker()
    pyautogui.mouseUp(button='left')
    pyautogui.mouseUp(button='right')
    time.sleep(0.1)
    hc.move((338, 594), 0.25)
    time.sleep(0.1)
    hc.click()
    time.sleep(3.3)
    hc.move((984, 524), 0.5)
    time.sleep(0.1)
    hc.click()
    time.sleep(0.1)
    hc.move((500, 320), 0.3)
    time.sleep(0.7)
    return


def select_beatmap(search_name):
    hc = pyclick.HumanClicker()
    pyautogui.mouseUp(button='left')
    pyautogui.mouseUp(button='right')
    time.sleep(0.5)
    for letter in search_name:
        win32api.keybd_event(KEY_DICT[letter], 0, 0, 0)
        time.sleep(0.04)
        win32api.keybd_event(KEY_DICT[letter], 0, win32con.KEYEVENTF_KEYUP, 0)
    hc.move((830, 370), 2.5)
    time.sleep(0.8)
    hc.click()
    return


def launch_selected_beatmap():
    hc = pyclick.HumanClicker()
    time.sleep(0.2)
    hc.move((984, 524), 0.6)
    time.sleep(0.25)
    hc.click()
    time.sleep(0.3)
    hc.move((500, 320), 0.3)
    time.sleep(0.7)
    return


def enable_nofail():
    hc = pyclick.HumanClicker()
    time.sleep(0.2)
    hc.move((275, 590), 0.5)
    time.sleep(0.2)
    hc.click()
    time.sleep(0.2)
    hc.move((385, 195), 0.5)
    time.sleep(0.15)
    hc.click()
    time.sleep(0.2)
    hc.move((350, 520), 0.6)
    time.sleep(0.15)
    hc.click()
    time.sleep(0.3)
    hc.move((150, 520), 0.7)
    time.sleep(0.3)
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


def hide_chat(hc):
    time.sleep(0.05)
    hc.move((992, 619), 0.05)
    time.sleep(0.1)
    hc.click()
    time.sleep(0.05)
    hc.move((400, 220), 0.1)


def return_to_beatmap():
    hc = pyclick.HumanClicker()
    time.sleep(8)
    hc.move((800, 270), 0.8)
    time.sleep(0.2)
    hc.click()
    time.sleep(0.4)
    hc.click()
    hc.move((50, 605), 0.9)
    time.sleep(0.15)
    hc.click()
    time.sleep(0.4)
    return


def shut_annoying_window():
    hc = pyclick.HumanClicker()
    time.sleep(1)
    hc.move((1390, 782), 0.8)
    time.sleep(0.1)
    hc.click()
    return


def relocate():
    hc = pyclick.HumanClicker()
    time.sleep(0.1)
    hc.move((404, 312), 0.5)
    time.sleep(0.1)
    hc.click()
    return


## DEBUG
if __name__ == '__main__':
    process = start_osu()
    import utils.mouse_locator

    utils.mouse_locator.mouse_locator()
    # stop_osu(process)

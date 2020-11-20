import win32gui
import os
import subprocess
import psutil
import time

from utils.osu_config import OSU_FOLDER_PATH, swap_to_AI, restore_player, USERNAME, AI_NAME

def start_osu():
    try:
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

##DEBUG
if __name__ == '__main__':
    osu_process = start_osu()
    time.sleep(20)
    stop_osu(osu_process)
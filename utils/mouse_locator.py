import pyautogui
import time


def mouse_locator():
    while True:
        time.sleep(0.5)
        x, y = pyautogui.position()
        print('(' + str(x) + ', ' + str(y) + ')')


if __file__ == '__main__':
    mouse_locator()


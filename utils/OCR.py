import time

import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms
import torch


import utils.screen
import utils.osu_routines

SCORE_REGION = (871, 26, 1024, 54)
ACCURACY_REGION = (916, 61, 1024, 87) # OBSOLETE
HIDE_CHAT_REGION = (971, 611, 1018, 626)


def invert_convert(screen):
    img = torchvision.transforms.ToPILImage()(screen.detach().permute(2, 0, 1).cpu()).convert('L')
    return ImageOps.invert(img)


def get_score(screen):
    tmp = screen.screenshot(region=SCORE_REGION)
    img = invert_convert(tmp).point(lambda x: 0 if x < 110 else x, 'L')
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    if s == '\x0c':
        return -1  # ToDo: do something about this
    else:
        s = int(s)
        # if s > 10000:
        #     img.save('truc' + str(s) + '.png')
        return s


def get_accuracy(screen):
    img = utils.screen.get_screen_region(screen, ACCURACY_REGION)
    img = invert_convert(img)
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=,0123456789')
    s = s.split(',')
    return float(s[0] + '.' + s[1])


def check_stuck_social(screen):
    img = utils.screen.get_screen_region(screen, HIDE_CHAT_REGION)
    img = invert_convert(img)
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=HideChathSow')
    return s[:8] == "HideChat"


if __name__ == '__main__':
    pass
    # process = utils.osu_routines.start_osu()
    # time.sleep(20)
    # screen = utils.screen.init_screen()
    # check_stuck_social(screen)
    # stop_osu(process)

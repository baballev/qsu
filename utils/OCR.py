import pytesseract
from PIL import Image, ImageOps
import torchvision.transforms
import torch

import utils.screen

SCORE_REGION = (830, 26, 1024, 60)
ACCURACY_REGION = (916, 61, 1024, 87)
HIDE_CHAT_REGION = (971, 611, 993, 626)


def invert_convert(screen):
    img = torchvision.transforms.ToPILImage()(screen.detach().permute(2, 0, 1).cpu())
    return ImageOps.invert(img)


def get_score(screen):
    tmp = screen.screenshot(region=SCORE_REGION)
    img = invert_convert(tmp)
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    if s == '\x0c':
        return -1  # ToDo: do something about this
    else:
        return int(s)


def get_accuracy(screen):
    img = utils.screen.get_screen_region(screen, ACCURACY_REGION)
    img = invert_convert(img)
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=,0123456789')
    s = s.split(',')
    return float(s[0] + '.' + s[1])


def check_stuck_social(screen):
    img = utils.screen.get_screen_region(screen, HIDE_CHAT_REGION)
    img = invert_convert(img)
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=hidechatsow')
    print(s)
    return s == "hide chat"


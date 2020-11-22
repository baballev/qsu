import pytesseract
from PIL import Image, ImageOps
import torchvision.transforms
import torch

import utils.screen

SCORE_REGION = (830, 34)
ACCURACY_REGION = (916, 62, 993)


def screen_to_score_img(screen):
    img = torchvision.transforms.ToPILImage()(screen.detach()[:SCORE_REGION[1], SCORE_REGION[0]:, :].permute(2, 0, 1).cpu())
    return ImageOps.invert(img)


def screen_to_accuracy_img(screen):
    print(screen.shape)
    img = torchvision.transforms.ToPILImage()(screen.detach()[SCORE_REGION[1]+1:ACCURACY_REGION[1], ACCURACY_REGION[0]:ACCURACY_REGION[2], :].permute(2, 0, 1).cpu())
    return ImageOps.invert(img)


def get_score(screen): #ToDO: Handle bad reading?
    img = screen_to_score_img(utils.screen.get_screen(screen))
    #img.save('temp.png')
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    if s == '\x0c':
        return -1 # ToDo: do something about this
    else:
        return int(s)


def get_accuracy(screen):
    img = screen_to_accuracy_img(utils.screen.get_screen(screen))
    #img.save('temp2.png')
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=,0123456789')
    s = s.split(',')
    return float(s[0] + '.' + s[1])

def is_terminal_state(screen):
    pass
import time
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torch.utils.data
import win32gui

import utils.screen
import utils.osu_routines

SCORE_REGION = (875, 27, 1019, 54)
ACCURACY_REGION = (916, 61, 1024, 87)  # OBSOLETE
HIDE_CHAT_REGION = (971, 611, 1018, 626)


'''
def invert_convert(screen):
    img = torchvision.transforms.ToPILImage()(screen.detach().permute(2, 0, 1).cpu()).convert('L')
    return ImageOps.invert(img)


def get_score(screen):
    tmp = screen.screenshot(region=SCORE_REGION)
    img = invert_convert(tmp).point(lambda x: 0 if x < 110 else x, 'L')
    s = pytesseract.image_to_string(img, config='--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789')
    if s == '\x0c':
        return -1  # ToDo: do something about this
    else:
        torchvision.transforms.ToPILImage()(tmp.detach().permute(2, 0, 1).cpu()).save(s.split('\n\x0c')[0] + '.png')
        # if s > 10000:
        #     img.save('truc' + str(s) + '.png')
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
    s = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=HideChathSow')
    return s[:8] == "HideChat"
'''


class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.paths = [os.path.join(directory, f) for f in os.listdir(directory)]
        self.length = len(self.paths)*8

    def __getitem__(self, index):
        i, j = divmod(index, 8)
        img = Image.open(self.paths[i])
        t = int(self.paths[i].split('/')[-1][j])

        img = torchvision.transforms.ToTensor()(img)[:, :, j*18:(j+1)*18]
        return img, t

    def __len__(self):
        return self.length


class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__() # input: 18 width, 27 h
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(2,2)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = int((conv2d_size_out(int((conv2d_size_out(18) - 2)/2 +1)) - 2)/2 + 1)
        convh = int((conv2d_size_out(int((conv2d_size_out(27) - 2)/2 +1)) - 2)/2 + 1)

        self.fc5 = nn.Linear(convw*convh, 10)

    def forward(self, x):
        x = F.relu(self.pool2(self.conv1(x)))
        x = F.relu(self.pool4(self.conv3(x)))
        x = self.fc5(x.view(x.size(0), -1))
        return x

    def train(self, weights_save='OCR_digit.pt'):
        dataset = ScoreDataset('E:/Programmation/Python/qsu!/dataset/')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
        valid = ScoreDataset('E:/Programmation/Python/qsu!/valid/')
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=10, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        i = 0
        previous_loss = 15000000.0
        for _ in range(15):
            for data, j in dataloader:
                optimizer.zero_grad()
                out = self.forward(data)
                loss = F.cross_entropy(out, j)
                loss.backward()
                optimizer.step()
                i += 1

            with torch.no_grad():
                running_loss = 0.0
                for data, j in valid_loader:
                    loss = F.cross_entropy(self.forward(data), j)
                    running_loss += loss.item()

                current_loss = running_loss / len(valid)
                print(current_loss)

            if current_loss < previous_loss:
                previous_loss = current_loss
                torch.save(self.state_dict(), weights_save)
        return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_OCR(weights='./weights/OCR/OCR_digit.pt'):
    OCR = OCRModel().to(device)
    OCR.load_state_dict(torch.load(weights))
    return OCR


counter = 0


def get_score(screen, ocr, wndw):
    global counter
    with torch.no_grad():
        score_img = utils.screen.get_screen_region(screen, region=SCORE_REGION)
        if counter % 5 == 0:
            if not(score_img.sum()) or win32gui.GetWindowText(wndw) == 'osu!':
                return -1

        score_img = torch.stack([score_img[:, :, j*18:(j+1)*18] for j in range(8)], 0)
        if score_img.shape[2] > 27:
            return -1
        score_img = ocr(score_img)
        _, indices = torch.max(score_img, 1)
        s = 0.0
        for n, indic in enumerate(indices):
            s += indic * 10**(7-n)
        counter += 1
    return s.float()


if __name__ == '__main__':
    ## DATASET MAKER
    #utils.osu_routines.start_osu()
    #screen = utils.screen.init_screen()
    #while True:
    #   get_score(screen)
    #   time.sleep(0.5)

    ## TRAIN OCR MODEL
    #ocr = OCRModel()
    #ocr.train()

    ## GET_SCORE TESTS
    utils.osu_routines.start_osu()
    screen = utils.screen.init_screen()
    ocr = init_OCR('../weights/OCR/OCR_digit.pt')
    while True:
        time.sleep(2)
        sho = utils.screen.get_screen(screen)
        print(get_score(sho, ocr))


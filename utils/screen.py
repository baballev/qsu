from threading import Thread

import d3dshot
import time

import utils.osu_routines
import torchvision.transforms as transforms

WINDOW_REGION = (0, 26, 1024, 626)
GAME_REGION = (2, 26, 880, 626)
MANIA_REGION = (140, 26, 520, 626)
TAIKO_REGION = (50, 272, 650, 273)


def init_screen(capture_output="pytorch_float_gpu"):
    capturer = d3dshot.create(capture_output=capture_output, frame_buffer_size=10)
    print('Game Region capture size:')
    print(get_game_screen(capturer, skip_pixels=1).shape)
    print(get_game_screen(capturer, skip_pixels=4).shape)
    time.sleep(0.1)
    print('Mania Region capture size:')
    print(str(MANIA_REGION[2] - MANIA_REGION[0]) + ' x ' + str(MANIA_REGION[3] - MANIA_REGION[1]))
    print(str((MANIA_REGION[2] - MANIA_REGION[0]) // 4) + ' x ' + str((MANIA_REGION[3] - MANIA_REGION[1])//4))
    print('Taiko Region capture size:')
    print(str(TAIKO_REGION[2] - TAIKO_REGION[0]) + ' x ' + str(TAIKO_REGION[3] - TAIKO_REGION[1]))
    return capturer


def get_screen(capturer):
    return capturer.screenshot(region=WINDOW_REGION).permute(2, 0, 1)


def get_screen_region(capturer, region):
    return capturer.screenshot(region=region).permute(2, 0, 1)


def get_game_screen(capturer, skip_pixels=4):
    return capturer.screenshot(region=WINDOW_REGION).permute(2, 0, 1)[:, ::skip_pixels, ::skip_pixels]  # Divide each size by skip_pixels

counting = 0
def save_screen(capturer, output_dir, output_name, region=WINDOW_REGION, skip_pixels=None):
    global counting
    if skip_pixels is None:
        capturer.screenshot_to_disk(output_dir, output_name, region=region)
    else:
        tmp = capturer.screenshot(region=region).permute(2, 0, 1)[:, ::skip_pixels, ::skip_pixels]
        transforms.ToPILImage()(tmp).save(output_dir + output_name + str(counting) + '.png')

        counting += 1


## Testing new way to record
def start_capture(screen, fps=60):
    screen.capture(fps, region=TAIKO_REGION)


if __name__ == '__main__':
    counting = 0
    utils.osu_routines.start_osu()
    screen = init_screen()
    thread = Thread(target=screen.capture, args=(60, TAIKO_REGION))
    thread.start()
    time.sleep(20)
    prev_time = time.perf_counter()
    while True:
        time.sleep(0.025)
        while time.perf_counter() - prev_time < 1/30:
            pass
        prev_time = time.perf_counter()

        transforms.ToPILImage()(screen.get_latest_frame().permute(2, 0, 1)[2, ::4, ::4]).save('E:/Programmation/Python/qsu!/benchmark/07-JUN-2021/' + 'tmp' + str(counting) + '.png')
        counting += 1
        if counting == 250:

            t = screen.get_latest_frame().permute(2, 0, 1)[2, ::4, ::4]
            print(t)
            with open('E:/Programmation/Python/qsu!/benchmark/07-JUN-2021/tensor.txt', 'w') as f:
                f.write(str(t))
            break

import d3dshot
import time

import utils.osu_routines

WINDOW_REGION = (0, 26, 1024, 626)
GAME_REGION = (2, 26, 880, 626)
MANIA_REGION = (140, 26, 520, 626)


def init_screen(capture_output="pytorch_float_gpu"):
    print('Initiating screen capture.')
    capturer = d3dshot.create(capture_output=capture_output, frame_buffer_size=5)
    print(capturer.display)
    print('Game Region capture size:')
    print(get_game_screen(capturer, skip_pixels=1).shape)
    print(get_game_screen(capturer, skip_pixels=4).shape)
    time.sleep(0.1)
    print('Mania Region capture size:')
    print(str(MANIA_REGION[2] - MANIA_REGION[0]) + ' x ' + str(MANIA_REGION[3] - MANIA_REGION[1]))
    print(str((MANIA_REGION[2] - MANIA_REGION[0]) // 4) + ' x ' + str((MANIA_REGION[3] - MANIA_REGION[1])//4))
    return capturer


def get_screen(capturer):
    return capturer.screenshot(region=WINDOW_REGION).permute(2, 0, 1)


def get_screen_region(capturer, region):
    return capturer.screenshot(region=region).permute(2, 0, 1)


def get_game_screen(capturer, skip_pixels=4):
    return capturer.screenshot(region=WINDOW_REGION).permute(2, 0, 1)[:, ::skip_pixels, ::skip_pixels]  # Divide each size by skip_pixels


def save_screen(capturer, output_dir, output_name, region=WINDOW_REGION):
    capturer.screenshot_to_disk(output_dir, output_name, region=region)


if __name__ == '__main__':
    utils.osu_routines.start_osu()
    screen = init_screen()
    while True:
        time.sleep(5)
        save_screen(screen, './', 'tmp.png', region=GAME_REGION)

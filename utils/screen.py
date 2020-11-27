import d3dshot
import time
WINDOW_REGION = (0, 26, 1024, 626)


def init_screen(capture_output="pytorch_float_gpu"): # capture_output= 'pytorch_float_gpu'
    print('Initiating screen capture.')
    capturer = d3dshot.create(capture_output=capture_output, frame_buffer_size=5)
    print(capturer.display)
    return capturer


def get_screen(capturer):
    return capturer.screenshot(region=WINDOW_REGION).permute(2, 0, 1)


def get_screen_region(capturer, region):
    return capturer.screenshot(region=region)


def save_screen(capturer, output_dir, output_name):
    capturer.screenshot_to_disk(output_dir, output_name, region=WINDOW_REGION)


import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair_i
from track import TrackManager
from ba import bundle_adjust
from pair import process_dense_pair
import json

if __name__ == '__main__':
    config = json.load(open('main.json'))
    frames = load_frames(config['video'], scale_factor=config['scale_factor'], quiet=True)
    frames = select_key_frames(frames, drop=config['drop'])

    for i, frame in enumerate(frames):
        plt.figure()
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Кадр {i + 1} из {len(frames)}')
        plt.show()
    #t, acc, gyr, mag = load_imu()
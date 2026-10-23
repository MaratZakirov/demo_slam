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
import csv

if __name__ == '__main__':
    # load configs
    main_config = json.load(open('main.json'))
    camera_config = json.load(open(main_config['camera_json']))

    # load IMU data
    t   = np.loadtxt(main_config['imu']['t'])
    acc = np.loadtxt(main_config['imu']['acc'], delimiter=',')
    gyr = np.loadtxt(main_config['imu']['gyr'], delimiter=',')
    mag = np.loadtxt(main_config['imu']['mag'], delimiter=',')

    # load and trim camera parameters
    dist = np.array(camera_config['D'])
    K = main_config['scale_factor'] * np.array(camera_config['K'])
    K[2, 2] = 1

    # load frames
    frames = load_frames(main_config['video'], scale_factor=main_config['scale_factor'], quiet=True)
    frames = select_key_frames(frames, drop=main_config['drop'])
    frames = undistort_frames(frames, dist, K)

    for i, frame in enumerate(frames):
        plt.figure()
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Кадр {i + 1} из {len(frames)}')
        plt.show()


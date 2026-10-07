import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair

if __name__ == '__main__':
    video_path = 'sample/kitchen.mp4'
    frames = load_frames(video_path, quiet=True)

    # reduce frame num
    frames = frames[::10]

    # trunkate
    frames = frames[:5]

    # reduce size
    for i in range(len(frames)):
        frames[i] = get_center_crop_coords(frames[i])

    # process frames
    data = []
    for i in range(0, len(frames)-1):
        frame_i  = frames[i]
        frame_i1 = frames[i + 1]

        R_i, t_i, key_pts_2d_i, key_pts_3d_i = process_stereo_pair(
            frame_i, frame_i1,
            data[-1][2] if i > 0 else np.zeros((0, 2)),
            data[-1][3] if i > 0 else np.zeros((0, 3)),
            attributes=None, num=i, full_mode=False, verbose=0)[:4]

        data.append([R_i, t_i, key_pts_2d_i, key_pts_3d_i])

    # Cascade correction for camera 0
    # R_0toi and t_0toi describes how to convert coordinate system 0 to i
    R_0toi = np.eye(3)
    t_0toi = np.zeros((3, 1))

    # At the end
    # all 3d points in data will be in coordinate system of camera 0
    # all poses will be in terms of coordinate system of camera 0
    for i in range(len(data)):
        data[i][3] = (R_0toi.T @ (data[i][3].T - t_0toi)).T

        # make iteration
        R_i = data[i][0]
        t_i = data[i][1]
        data[i][0] = R_0toi.T
        data[i][1] = -R_0toi.T @ t_0toi
        R_0toi = R_i @ R_0toi
        t_0toi = R_i @ t_0toi + t_i

    # build scene
    K = get_matrix_K_from_frame(frames[0])
    visualize_slam_scene(data, K=K)

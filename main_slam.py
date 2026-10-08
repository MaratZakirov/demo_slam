import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair_i
from track import TrackManager
from ba import bundle_adjust

if __name__ == '__main__':
    video_path = 'sample/kitchen.mp4'
    frames = load_frames(video_path, quiet=True)
    frames = frames[::10]
    frames = frames[:10]
    for i in range(len(frames)):
        frames[i] = get_center_crop_coords(frames[i])

    tm = TrackManager()
    cam_poses = []    # [R_i, t_i] — позы cam_i → cam_{i+1}

    # Накопитель: cam_left → world (frame 0)
    R_0toi = np.eye(3)
    t_0toi = np.zeros((3, 1))

    for i in range(len(frames) - 1):
        R_i, t_i = process_stereo_pair_i(
            frames[i], frames[i+1], tm,
            world_pose=(R_0toi.T, -R_0toi.T @ t_0toi), # R_ito0 t_it0
            cam_i=i, cam_ip1=i + 1, verbose=0)

        # instantly obtaining camera pose in camera 0 system
        cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])

        # Обновляем: 0→(i+1) = (i→i+1) ∘ (0→i)
        R_0toi = R_i @ R_0toi
        t_0toi = R_i @ t_0toi + t_i

        print(f"[iter {i}] {tm.stats(i)}")

    # last camera
    cam_poses.append([R_0toi.T, -R_0toi.T @ t_0toi])

    K = get_matrix_K_from_frame(frames[0])

    cam_poses, tm, result = bundle_adjust(tm, cam_poses, K)

    visualize_slam_scene(tm, cam_poses, K=K)
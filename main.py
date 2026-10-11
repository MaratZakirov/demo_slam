import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair_i
from track import TrackManager
from ba import bundle_adjust
from pair import process_dense_pair

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

    visualize_slam_scene(tm, cam_poses, K=K, color_mode='pixel')
    visualize_slam_scene(tm, cam_poses, K=K, color_mode='camera')

    # Select some poses
    candidates = [(i, i+1) for i in range(0, len(frames)-1, 2)]

    ordered_pairs = []
    for i, j in candidates:
        R_ji, t_ji = relative_pose(cam_poses[i], cam_poses[j])
        ordered_pairs.append((i, j) if t_ji[0, 0] < 0 else (j, i))   # i — left, j — right

    all_pts = []
    all_cols = []
    for left_idx, right_idx in ordered_pairs:
        X_world, cols = process_dense_pair(
            frame_l=frames[left_idx],
            frame_r=frames[right_idx],
            pose_l=cam_poses[left_idx],
            pose_r=cam_poses[right_idx],
            K=K,
            pair_id=(left_idx, right_idx),
        )
        all_pts.append(X_world)
        all_cols.append(cols)
        print(f"[pair {left_idx}-{right_idx}] +{len(X_world)} pts")

    all_pts  = np.vstack(all_pts)
    all_cols = np.vstack(all_cols)
    print(f"[dense] total points: {len(all_pts)}")

    save_ply_binary("dense.ply", all_pts, all_cols)
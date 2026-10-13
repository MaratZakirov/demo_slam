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
    FILTER = True
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

    # filter mode requires much more data
    if FILTER:
        candidates = (candidates
                      + [(i, i+2) for i in range(0, len(frames)-2, 2)]
                      + [(i, i+2) for i in range(1, len(frames)-2, 2)]
                      + [(i, i+3) for i in range(0, len(frames)-2, 3)])

    ordered_pairs = []
    for i, j in candidates:
        R_ji, t_ji = relative_pose(cam_poses[i], cam_poses[j])
        ordered_pairs.append((i, j) if t_ji[0, 0] < 0 else (j, i))   # i — left, j — right

    all_pts = []
    all_cols = []
    all_i = []
    for i, (left_idx, right_idx) in enumerate(ordered_pairs):
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
        all_i.append(np.full(len(cols), i, dtype=np.int16))
        print(f"[pair {left_idx}-{right_idx}] +{len(X_world)} pts")

    all_pts  = np.vstack(all_pts)
    all_cols = np.vstack(all_cols)
    all_i = np.concatenate(all_i)
    print(f"[dense] total points: {len(all_pts)}")

    if FILTER:
        # save raw points for voting
        np.save("X3d.npy",  all_pts)
        np.save("cols.npy", all_cols)
        np.save("pair.npy", all_i)
        print("[dense] cached to X3d.npy / cols.npy / pair.npy")

    save_ply_binary("dense.ply", all_pts, all_cols)

    if FILTER:
        # load raw points for voting
        all_pts = np.load("X3d.npy")
        all_cols = np.load("cols.npy")
        all_i = np.load("pair.npy")

        # perform voxel filtering
        vox_xyz, vox_rgb = voxel_vote(all_pts, all_cols, all_i, voxel_size=0.05, min_count=2, min_pairs=2)
        save_ply_binary("dense_voxel.ply", vox_xyz, vox_rgb)
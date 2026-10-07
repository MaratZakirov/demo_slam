import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt
from pair import process_stereo_pair

if __name__ == '__main__':
    video_path = 'sample/kitchen.mp4'
    frames = load_frames(video_path, quiet=True)
    frames = frames[::10]

    frame_a = get_center_crop_coords(frames[2])
    frame_b = get_center_crop_coords(frames[1])
    frame_c = get_center_crop_coords(frames[0])

    print('=== pair_ab ===')
    R_ab, t_ab, key_pts_2d_ab, key_pts_3d_ab, dense_pts_3d_ab, dense_pts_colors_ab, dense_pts_conf_ab \
        = process_stereo_pair(frame_a, frame_b, num=0, verbose=0)

    print('=== pair_bc ===')
    attr = {}
    R_bc, t_bc, key_pts_2d_bc, key_pts_3d_bc, dense_pts_3d_bc, dense_pts_colors_bc, dense_pts_conf_bc \
        = process_stereo_pair(frame_b, frame_c, key_pts_2d_ab, key_pts_3d_ab, attributes=attr, num=1, verbose=0)

    print('=== correct R_ab t_ab ===')
    M, scale = cv2.estimateAffine3D(attr['pts_3d_curr'], attr['pts_3d_prev'], force_rotation=True)
    R_ab, t_ab = M[:, :3].T, -M[:, 3:4]
    print(f'Correction applied scale {scale}')

    show_3d_match(key_pts_3d_ab, key_pts_3d_bc @ R_ab - t_ab.T)
    show_3d_match(attr['pts_3d_prev'], attr['pts_3d_curr'] @ R_ab - t_ab.T)

    # Тест на включенность
    print('Тест на включенность:', (np.sum(np.abs(key_pts_3d_ab[:, None] - attr['pts_3d_prev'][None]), axis=2) == 0).sum(), min(len(key_pts_3d_ab), len(attr['pts_3d_prev'])))

    # Тест на невязку
    print('Тест на невязку:', np.median(np.abs((attr['pts_3d_prev'] - (attr['pts_3d_curr'] @ R_ab - t_ab.T)))))

    print('=== fuse ===')
    dense_pts_3d_fused  = np.concatenate([dense_pts_3d_ab, dense_pts_3d_bc @ R_ab - t_ab.T], axis=0)
    dense_pts_colors_fused = np.concatenate([dense_pts_colors_ab, dense_pts_colors_bc], axis=0)

    save_ply_fast_numpy('myscene_ab.ply', dense_pts_3d_ab.reshape(-1, 3), dense_pts_colors_ab.reshape(-1, 3))
    save_ply_fast_numpy('myscene_bc.ply', dense_pts_3d_bc.reshape(-1, 3), dense_pts_colors_bc.reshape(-1, 3))
    save_ply_fast_numpy('fused.ply', dense_pts_3d_fused.reshape(-1, 3), dense_pts_colors_fused.reshape(-1, 3))
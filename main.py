import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt

# Consider only Left - Right stereo-pairs
def process_stereo_pair(frame_left, frame_right, pts_l_prev=np.zeros((0, 2)), pts_3d_prev=np.zeros((0, 3)), attributes=None, verbose=True, full_mode=True, num: int = -1):
    K = get_matrix_K_from_frame(frame_left)

    matched_pts_l, matched_pts_r, pts_r_prev, mask_prev = get_matches_using_optical_flow(frame_left, frame_right, pts_l_prev=pts_l_prev)

    # filtering previous data
    pts_l_prev = pts_l_prev[mask_prev]
    pts_r_prev = pts_r_prev[mask_prev]
    pts_3d_prev = pts_3d_prev[mask_prev]

    E, mask = cv2.findEssentialMat(
        np.concatenate([matched_pts_l, pts_l_prev], axis=0),
        np.concatenate([matched_pts_r, pts_r_prev], axis=0),
        cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=0.5)

    points, R, t, mask_pose = cv2.recoverPose(
        E,
        np.concatenate([matched_pts_l, pts_l_prev], axis=0),
        np.concatenate([matched_pts_r, pts_r_prev], axis=0),
        cameraMatrix=K, mask=mask)

    assert t[0, 0] < 0, 'Reorder frames from left to right'

    if verbose:
        plot_flow_vectors(frame_left, matched_pts_l, matched_pts_r, max_arrows=700)
        plot_matches(frame_left, frame_right, matched_pts_l, matched_pts_r)

    N = len(matched_pts_l)
    valid_mask    = mask_pose.ravel() == 1
    pts_l_inliers = matched_pts_l[:N][valid_mask[:N]]
    pts_r_inliers = matched_pts_r[:N][valid_mask[:N]]

    # filtering previous data
    pts_l_prev = pts_l_prev[valid_mask[N:]]
    pts_r_prev = pts_r_prev[valid_mask[N:]]
    pts_3d_prev = pts_3d_prev[valid_mask[N:]]

    #if verbose:
    #    print(f'[pair] matched={len(matched_pts_l)}  inliers={valid_mask.sum()}  '
    #          f'inherited_among_inliers={inherited.sum()}')

    # use base corretion
    if len(pts_3d_prev) >= 3:
        pts_4d_curr = cv2.triangulatePoints(
            K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
            K @ np.hstack([R, t.reshape(3, 1)]),
            pts_l_prev.T, pts_r_prev.T)
        pts_3d_curr = (pts_4d_curr[:3] / pts_4d_curr[3]).T

        # TODO I have questions for this...
        scale = np.median(np.abs(pts_3d_prev[:, 2])) / np.median(np.abs(pts_3d_curr[:, 2]))

        t = t * scale

        if verbose:
            print(f'[pair] z_prev={np.median(np.abs(pts_3d_prev[:, 2])):.4f} z_curr={np.median(np.abs(pts_3d_curr[:, 2])):.4f} scale={scale:.4f}')

        # Recalculate with correct t
        if not attributes is None:
            pts_4d_curr = cv2.triangulatePoints(
                K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
                K @ np.hstack([R, t.reshape(3, 1)]),
                pts_l_prev.T, pts_r_prev.T)
            pts_3d = (pts_4d_curr[:3] / pts_4d_curr[3]).T
            attributes['pts_3d_curr'] = pts_3d
            attributes['pts_3d_prev'] = pts_3d_prev

    # now we have correct t, so we can traingulate new current points
    pts_4d = cv2.triangulatePoints(
        K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
        K @ np.hstack([R, t.reshape(3, 1)]),
        pts_l_inliers.T, pts_r_inliers.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T

    # Return R, t, + key points on right camera (will be left for next stereo-pair)
    # calculated points 3d for these keypoints
    if not full_mode:
        return R, t, pts_r_inliers, pts_3d, np.zeros((0, 3))

    # =====================#
    # Build dense 3D model #
    # =====================#

    if verbose:
        plot_flow_vectors(frame_left, pts_l_inliers, pts_r_inliers, max_arrows=700)
        plot_matches(frame_left, frame_right, pts_l_inliers, pts_r_inliers)
        visualize_scene_matplotlib(pts_l_inliers, pts_r_inliers, K, R, t)

    height, width = frame_left.shape[:2]
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=dist_coeffs,
        cameraMatrix2=K, distCoeffs2=dist_coeffs,
        imageSize=(width, height), R=R, T=t.reshape(3, 1))
    assert min(roi1) >= 0

    map1_x, map1_y = cv2.initUndistortRectifyMap(K, None, R1, P1, (width, height), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K, None, R2, P2, (width, height), cv2.CV_32FC1)

    rectified_frame_l = cv2.remap(frame_left, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_frame_r = cv2.remap(frame_right, map2_x, map2_y, cv2.INTER_LINEAR)

    if verbose:
        imshow('Rectified frames', np.concatenate([rectified_frame_l, rectified_frame_r], axis=1))

    try:
        disp = np.load(f"disp_{num}.npy")
        print("Disparity loaded from cache")
    except FileNotFoundError:
        disp = HitNetDisparity()(rectified_frame_l, rectified_frame_r)
        print("Disparity calculated")
        if num >= 0:
            np.save(f"disp_{num}.npy", disp)
            print("Disparity saved to cache")

    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    disp_visual = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if verbose:
        imshow("Disparity Map", disp_visual)

    dense_pts_3d = cv2.reprojectImageTo3D(disp, Q)
    dense_pts_conf = np.ones_like(disp)

    percentile = 4
    lo_bound = np.percentile(dense_pts_3d[..., 2], percentile)
    up_bound = np.percentile(dense_pts_3d[..., 2], 100 - percentile)

    dense_pts_colors = cv2.cvtColor(rectified_frame_l, cv2.COLOR_BGR2RGB)
    mask = (disp > 0.02) & (disp < 120) & np.isfinite(dense_pts_3d[..., 2]) & \
           (dense_pts_3d[..., 2] < up_bound) & (dense_pts_3d[..., 2] > lo_bound)
    roi_mask = np.zeros_like(mask)
    roi_mask[roi1[1]:roi1[1] + roi1[3], roi1[0]:roi1[0] + roi1[2]] = True
    mask = mask & roi_mask

    dense_pts_3d[~mask]     = 0
    dense_pts_colors[~mask] = 0
    dense_pts_conf[~mask]   = 0

    # send dense points back to original left frame coordinate system
    dense_pts_3d = (dense_pts_3d.reshape(-1, 3) @ R1).reshape(*dense_pts_3d.shape)

    if verbose:
        save_ply_fast_numpy('myscene.ply', dense_pts_3d, dense_pts_colors)

    return R, t, pts_r_inliers, pts_3d, dense_pts_3d, dense_pts_colors, dense_pts_conf

if __name__ == '__main__':
    video_path = 'sample/kitchen.mp4'
    frames = load_frames(video_path, quiet=True)
    frames = frames[::10]

    frame_a = get_center_crop_coords(frames[2])
    frame_b = get_center_crop_coords(frames[1])
    frame_c = get_center_crop_coords(frames[0])

    print('=== pair_ab ===')
    R_ab, t_ab, key_pts_2d_ab, key_pts_3d_ab, dense_pts_3d_ab, dense_pts_colors_ab, dense_pts_conf_ab \
        = process_stereo_pair(frame_a, frame_b, num=0, verbose=False)

    print('=== pair_bc ===')
    attr = {}
    R_bc, t_bc, key_pts_2d_bc, key_pts_3d_bc, dense_pts_3d_bc, dense_pts_colors_bc, dense_pts_conf_bc \
        = process_stereo_pair(frame_b, frame_c, key_pts_2d_ab, key_pts_3d_ab, attributes=attr, num=1, verbose=False)

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

    save_ply_fast_numpy('fused.ply', dense_pts_3d_fused.reshape(-1, 3), dense_pts_colors_fused.reshape(-1, 3))
import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt

# Consider only Left - Right stereo-pairs
def process_stereo_pair(frame_left, frame_right, pts_l_prev=np.zeros((0, 2)), pts_3d_prev=np.zeros((0, 3)), attributes=None, verbose=True, full_mode=True, num: int = -1):
    M_in = len(pts_l_prev)
    assert len(pts_3d_prev) == M_in, \
        f"pts_l_prev and pts_3d_prev must have same length: {M_in} vs {len(pts_3d_prev)}"
    K = get_matrix_K_from_frame(frame_left)

    matched_pts_l, matched_pts_r, pts_r_prev, mask_prev = get_matches_using_optical_flow(frame_left, frame_right, pts_i_inherited=pts_l_prev)

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

    if full_mode:
        assert t[0, 0] < 0, 'Reorder frames from left to right'

    if verbose:
        plot_flow_vectors(frame_left, matched_pts_l, matched_pts_r, max_arrows=700)
        plot_matches(frame_left, frame_right, matched_pts_l, matched_pts_r)

    N = len(matched_pts_l)
    valid_mask    = mask_pose.ravel() == 1
    pts_l_inliers = matched_pts_l[:N][valid_mask[:N]]
    pts_r_inliers = matched_pts_r[:N][valid_mask[:N]]

    combined_prev_mask = np.zeros(M_in, dtype=bool)
    combined_prev_mask[mask_prev] = valid_mask[N:]

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
        return (R, t,
                pts_l_inliers,  # 2D новых на кадре i
                pts_r_inliers,  # 2D новых на кадре i+1
                pts_r_prev,     # ← 2D унаследованных на кадре i+1 (выжившие)
                combined_prev_mask,  # ← какие из входа дожили
                pts_3d,  # 3D новых (в системе кадра i)
                np.zeros((0, 0, 3)), np.zeros((0, 0, 3)), np.zeros((0, 0, 1)))

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

    return (R, t,
            pts_l_inliers, pts_r_inliers,
            pts_r_prev, combined_prev_mask,
            pts_3d,
            dense_pts_3d, dense_pts_colors, dense_pts_conf)

# Second version w/o HitNet
# process frames i -> i+1 where also i-th is Left and i+1 is Right
# regardless of their real motion
# HitNet and dense stereo map will be applied in the next stage
def process_stereo_pair_i(frame_i, frame_ip1, tm, cam_i, cam_ip1, world_pose, verbose=False):
    # Пара (frame_i, frame_ip1) == (cam_i, cam_ip1).
    # Всё состояние треков живёт в `tm`. Пара его обновляет.
    #
    # world_pose: (R_ito0, t_ito0), поза cam_i в системе кадра 0:
    # Нужна для перевода 3D новых точек из системы cam_left в систему кадра 0
    #
    # Returns: R, t — поза cam_ip1 относительно cam_i (X_ip1 = R @ X_i + t).
    K = get_matrix_K_from_frame(frame_i)

    # ---- Активные треки с их 2D на кадре cam_left ----
    active_ids, pts_i_inherited, X_i_inhereted = tm.active_on(cam_i)

    # ---- LK: тянем их на frame_right ----
    pts_i_new, pts_ip1_new, pts_ip1_inherited, mask_prev = \
        get_matches_using_optical_flow(frame_i, frame_ip1,
                                       pts_i_inherited=pts_i_inherited)

    active_ids = active_ids[mask_prev]
    pts_i_inherited = pts_i_inherited[mask_prev]
    pts_ip1_inherited = pts_ip1_inherited[mask_prev]
    X_i_inhereted = X_i_inhereted[mask_prev]

    # ---- Essential + RANSAC по всем (новым + унаследованным) ----
    E, mask = cv2.findEssentialMat(
        np.concatenate([pts_i_new, pts_i_inherited], axis=0),
        np.concatenate([pts_ip1_new, pts_ip1_inherited], axis=0),
        cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=0.5)

    _, R, t, mask_pose = cv2.recoverPose(
        E,
        np.concatenate([pts_i_new, pts_i_inherited], axis=0),
        np.concatenate([pts_ip1_new, pts_ip1_inherited], axis=0),
        cameraMatrix=K, mask=mask)

    N = len(pts_i_new)
    valid_mask = mask_pose.ravel() == 1

    # Новые точки
    pts_i_new = pts_i_new[:N][valid_mask[:N]]
    pts_ip1_new = pts_ip1_new[:N][valid_mask[:N]]

    # Унаследованные, дожившие до конца
    surv = valid_mask[N:]
    alive_ids = active_ids[surv]
    alive_pts_i_inherited = pts_i_inherited[surv]
    alive_pts_ip1_inherited = pts_ip1_inherited[surv]
    alive_X_i_inherited = X_i_inhereted[surv] # 3D inherited from previous cameras which are already in world (camera 0) system

    # ---- Scale correction ----
    # essential-matrix даёт ||t||=1. Масштаб восстанавливаем, сравнивая
    # глубину унаследованных точек в системе cam_left (из их X_world)
    # с глубиной, которую даёт свежая триангуляция по текущей паре.
    if len(alive_ids) >= 3:
        R_ito0, t_ito0 = world_pose
        z_inherited_world = np.median(np.abs(alive_X_i_inherited[:, 2]))
        X_4d = cv2.triangulatePoints(
            K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
            K @ np.hstack([R, t.reshape(3, 1)]),
            alive_pts_i_inherited.T, alive_pts_ip1_inherited.T)
        X_i_inherited_cam_i = (X_4d[:3] / X_4d[3]).T

        # convert retriangulated inherited points in world (camera 0) system
        X_i_inherited_cam_i = (R_ito0 @ X_i_inherited_cam_i.T + t_ito0).T   # 4.35

        z_inherited_cam_i = np.median(np.abs(X_i_inherited_cam_i[:, 2]))
        scale = z_inherited_world / z_inherited_cam_i
        t = t * scale

        median_shift = np.median(np.linalg.norm(np.abs(X_i_inherited_cam_i - alive_X_i_inherited), axis=1))

        if 1:#verbose:
            print(f"[pair {cam_i}] scale={scale:.4f} z_world={z_inherited_world:.3f} z_curr={z_inherited_cam_i:.3f} median_shift={median_shift:.3f}")

    # ---- 5. Триангуляция новых точек в системе cam_left ----
    if len(pts_i_new) > 0:
        X_4d = cv2.triangulatePoints(
            K @ np.hstack([np.eye(3), np.zeros((3, 1))]),
            K @ np.hstack([R, t.reshape(3, 1)]),
            pts_i_new.T, pts_ip1_new.T)
        X_i_cam_i = (X_4d[:3] / X_4d[3]).T
    else:
        X_i_cam_i = np.zeros((0, 3))

    # ---- TrackManager ----
    # Продлеваем выживших на cam_right
    tm.extend(alive_ids, cam_ip1, alive_pts_ip1_inherited)

    # Рождаем новых на cam_left, сразу продлеваем на cam_right
    new_ids = tm.birth(cam_i, pts_i_new)
    tm.extend(new_ids, cam_ip1, pts_ip1_new)

    # Записываем 3D новых (переводим cam_left → world)
    if len(new_ids) > 0:
        R_ito0, t_ito0 = world_pose
        t_ito0 = np.asarray(t_ito0).reshape(3, 1)
        tm.set_X(new_ids, (R_ito0 @ X_i_cam_i.T).T + t_ito0.ravel())

    # ---- Debug ----
    if verbose:
        plot_flow_vectors(frame_i, pts_i_new, pts_ip1_new, max_arrows=700)
        plot_matches(frame_i, frame_ip1, pts_i_new, pts_ip1_new)

    return R, t
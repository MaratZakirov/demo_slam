import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt

# Second version w/o HitNet
# process frames i -> i+1 where also i-th is Left and i+1 is Right
# regardless of their real motion
# HitNet and dense stereo map will be applied in the next stage
def process_stereo_pair_i(frame_i, frame_ip1, tm, cam_i, cam_ip1, world_pose, K=None, H_iip1=None, verbose=False):
    # Пара (frame_i, frame_ip1) == (cam_i, cam_ip1).
    # Всё состояние треков живёт в `tm`. Пара его обновляет.
    #
    # world_pose: (R_ito0, t_ito0), поза cam_i в системе кадра 0:
    # Нужна для перевода 3D новых точек из системы cam_left в систему кадра 0
    #
    # Returns: R, t — поза cam_ip1 относительно cam_i (X_ip1 = R @ X_i + t).
    if K is None:
        K = get_matrix_K_from_frame(frame_i)

    # ---- Активные треки с их 2D на кадре cam_left ----
    active_ids, pts_i_inherited, X_i_inhereted = tm.active_on(cam_i)

    # ---- LK: тянем их на frame_right ----
    pts_i_new, pts_ip1_new, pts_ip1_inherited, mask_prev = \
        get_matches_using_optical_flow(frame_i, frame_ip1,
                                       pts_i_inherited=pts_i_inherited,
                                       H=H_iip1)

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

        if verbose:
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

    colors = None
    if len(pts_i_new) > 0:
        h, w = frame_i.shape[:2]
        colors = frame_i[
            pts_i_new[:, 1].round().astype(int).clip(0, h - 1),
            pts_i_new[:, 0].round().astype(int).clip(0, w - 1), ::-1
        ].copy()

    # Рождаем новых на cam_left, сразу продлеваем на cam_right
    new_ids = tm.birth(cam_i, pts_i_new, colors=colors)
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

def process_dense_pair(frame_l, frame_r, pose_l, pose_r, K,
                       pair_id=(0, 0), verbose=False):
    """
    Плотное стерео для пары (left, right) с известными позами в world (frame 0).
    Возвращает (X_world, colors_rgb): (N, 3) float64 в world, (N, 3) uint8 RGB.
    """
    R_l, t_l = pose_l
    R_r, t_r = pose_r

    R_lr, t_lr = relative_pose(pose_l, pose_r)   # X_r = R_lr @ X_l + t_lr
    assert t_lr[0, 0] < 0, f"pair {pair_id} is not left-right"

    height, width = frame_l.shape[:2]
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=dist_coeffs,
        cameraMatrix2=K, distCoeffs2=dist_coeffs,
        imageSize=(width, height),
        R=R_lr, T=t_lr.reshape(3, 1))

    map1_x, map1_y = cv2.initUndistortRectifyMap(K, None, R1, P1, (width, height), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K, None, R2, P2, (width, height), cv2.CV_32FC1)

    rect_l = cv2.remap(frame_l, map1_x, map1_y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(frame_r, map2_x, map2_y, cv2.INTER_LINEAR)

    disp = HitNetDisparity()(rect_l, rect_r)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        show_disparity(disp, title=f'pair {pair_id}')

    dense_pts = cv2.reprojectImageTo3D(disp, Q)             # (H, W, 3) в rect-left
    colors    = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)     # (H, W, 3) uint8

    z = dense_pts[..., 2]
    lo = np.percentile(z, 4)
    hi = np.percentile(z, 96)

    roi_mask = np.zeros(z.shape, dtype=bool)
    roi_mask[roi1[1]:roi1[1] + roi1[3], roi1[0]:roi1[0] + roi1[2]] = True

    valid = (disp > 0.02) & (disp < 120) & roi_mask & \
            np.isfinite(z) & (z > 0) & (z > lo) & (z < hi)

    pts_rect = dense_pts[valid]                              # (N, 3) rect-left
    colors_v = colors[valid]                                 # (N, 3) RGB

    pts_l = pts_rect @ R1                                    # rect-left → orig-left
    X_world = (R_l @ pts_l.T).T + np.asarray(t_l).reshape(3) # orig-left → world

    if verbose:
        print(f"[dense {pair_id}] valid={valid.sum()}/{z.size}  "
              f"z_med={np.median(z[valid]):.3f}" if valid.any() else
              f"[dense {pair_id}] no valid pixels")

    return X_world, colors_v
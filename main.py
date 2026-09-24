import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt

# TODO we do not consider vertical stereopairs
def process_stereo_pair(frame_a, frame_b, prev_matches=None):
    K = get_matrix_K_from_frame(frame_a)

    while True:
        matched_pts_a, matched_pts_b = get_matches_using_optical_flow(frame_a, frame_b)

        # К — (Intrinsic Matrix)
        E, mask = cv2.findEssentialMat(
            matched_pts_a,
            matched_pts_b,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.5
        )

        # Фильтруем точки по маске RANSAC и раскладываем матрицу E на R и t
        points, R, t, mask_pose = cv2.recoverPose(E, matched_pts_a, matched_pts_b, cameraMatrix=K, mask=mask)

        # we work only with L - R pairs
        if t[0, 0] < 0:
            # plot optical flow
            plot_flow_vectors(frame_a, matched_pts_a, matched_pts_b, max_arrows=700)
            plot_matches(frame_a, frame_b, matched_pts_a, matched_pts_b)
            break
        else:
            assert False, 'Temporary assert reorder frames'
            # swap frames
            frame_a, frame_b = frame_b, frame_a
            print('Swap frames')

    valid_mask = mask.ravel() == 1
    pts_a_inliers = matched_pts_a[valid_mask]
    pts_b_inliers = matched_pts_b[valid_mask]

    # plot optical flow again
    plot_flow_vectors(frame_a, pts_a_inliers, pts_b_inliers, max_arrows=700)
    plot_matches(frame_a, frame_b, pts_a_inliers, pts_b_inliers)

    visualize_scene_matplotlib(pts_a_inliers, pts_b_inliers, K, R, t)

    height, width = frame_a.shape[:2]

    # (считаем, что линза идеальная)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K,
        distCoeffs1=dist_coeffs,
        cameraMatrix2=K,
        distCoeffs2=dist_coeffs,
        imageSize=(width, height),
        R=R,
        T=t.reshape(3, 1)
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(K, None, R1, P1, (width, height), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K, None, R2, P2, (width, height), cv2.CV_32FC1)

    # Получаем ректифицированные кадры
    rectified_frame_a = cv2.remap(frame_a, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_frame_b = cv2.remap(frame_b, map2_x, map2_y, cv2.INTER_LINEAR)

    imshow('Rectified frames', np.concatenate([rectified_frame_a, rectified_frame_b], axis=1))

    disp = HitNetDisparity()(rectified_frame_a, rectified_frame_b)

    # Чтобы увидеть её глазами, нормализуем для вывода на экран (0-255)
    # Замена для визуализации
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    disp_visual = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imshow("Disparity Map", disp_visual)

    points_3d = cv2.reprojectImageTo3D(disp, Q)
    conf = np.ones_like(disp)

    percentile = 4
    lo_bound = np.percentile(points_3d[..., 2], percentile)
    up_bound = np.percentile(points_3d[..., 2], 100 - percentile)

    colors = cv2.cvtColor(rectified_frame_a, cv2.COLOR_BGR2RGB)
    mask = (disp > 0.02) & (disp < 120) & (np.isfinite(points_3d[..., 2])) & (points_3d[..., 2] < up_bound) & (points_3d[..., 2] > lo_bound)

    points_3d[~mask] = 0
    colors[~mask]    = 0
    conf[~mask]      = 0

    # (N, 3)
    mask = points_3d[..., 2] > 0
    final_points = points_3d[mask]  # Массив размера (N, 3) с координатами X, Y, Z
    final_colors = colors[mask]     # Массив размера (N, 3) с цветами R, G, B

    save_ply_fast_numpy('myscene.ply', final_points, final_colors)

if __name__ == '__main__':
    video_path = 'sample/kitchen.mp4'
    frames = load_frames(video_path, quiet=True)
    frames = frames[::10]

    model_height = 720
    model_width  = 1280

    frame_a = get_center_crop_coords(frames[2])
    frame_b = get_center_crop_coords(frames[1])
    frame_c = get_center_crop_coords(frames[0])

    process_stereo_pair(frame_a, frame_b)
    #process_stereo_pair(frame_b, frame_c)
import cv2
import numpy as np
from utils import *
from debug import *
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def pca_spread(p):
    c = p - p.mean(0)
    cov = c.T @ c / max(len(c), 1)
    w = np.linalg.eigvalsh(cov)
    return np.sqrt(np.clip(w, 0, None))[::-1]


def cloud_distance(P_a, P_b, n_sample=5000):
    if len(P_a) == 0 or len(P_b) == 0:
        return 0.0, 0.0
    ia = np.random.choice(len(P_a), min(n_sample, len(P_a)), replace=False)
    ib = np.random.choice(len(P_b), min(n_sample, len(P_b)), replace=False)
    A = P_a[ia]; B = P_b[ib]
    d_ab, _ = cKDTree(B).query(A)
    d_ba, _ = cKDTree(A).query(B)
    d = np.concatenate([d_ab, d_ba])
    return float(d.mean()), float(np.median(d))


def analytical_transform(pair_ab, pair_bc, verbose=True):
    """Точное преобразование из СК cloud_bc в СК cloud_ab.

    Оба облака живут в ректифицированных СК первого кадра своей пары:
      cloud_ab — rect_cam_1 пары ab (= камера a с поворотом R1_ab)
      cloud_bc — rect_cam_1 пары bc (= камера b с поворотом R1_bc)

    Отношение камер a и b известно из recoverPose: X_b = R_ab @ X_a + t_ab.
    Отсюда:
      X_rect_bc = R1_bc @ R_ab @ R1_ab^T @ X_rect_ab + R1_bc @ t_ab
    Обратное (то, что применяем к cloud_bc, чтобы положить его в СК cloud_ab):
      R = R1_ab @ R_ab^T @ R1_bc^T
      t = -R @ (R1_bc @ t_ab) = -R1_ab @ R_ab^T @ t_ab
    """
    R_ab  = pair_ab['R'].astype(np.float64)          # 3x3
    t_ab  = pair_ab['t'].reshape(3).astype(np.float64)
    R1_ab = pair_ab['R1'].astype(np.float64)
    R1_bc = pair_bc['R1'].astype(np.float64)

    R_BtoA = R1_ab @ R_ab.T @ R1_bc.T
    t_BtoA = -R1_ab @ R_ab.T @ t_ab
    if verbose:
        print(f'[analytical] det(R)={np.linalg.det(R_BtoA):.4f}  '
              f'|t|={np.linalg.norm(t_BtoA):.4f}')
    return 1.0, R_BtoA, t_BtoA


def icp_refine(P_src, P_dst, s0, R0, t0, max_iter=10, subsample=20000,
               relative_thresh=0.02, min_matches=200, verbose=True):
    """Лёгкий ICP — только чтобы подчистить остаточную ошибку scale."""
    rng = np.random.default_rng(0)
    if len(P_src) > subsample:
        P_src = P_src[rng.choice(len(P_src), subsample, replace=False)]
    if len(P_dst) > subsample:
        P_dst = P_dst[rng.choice(len(P_dst), subsample, replace=False)]

    tree = cKDTree(P_dst)
    extent = np.linalg.norm(P_dst.max(0) - P_dst.min(0))
    max_dist = extent * relative_thresh
    s, R, t = float(s0), R0.copy(), t0.copy()

    for it in range(max_iter):
        P_cur = s * (P_src @ R.T) + t
        dist, idx = tree.query(P_cur)
        m = dist < max_dist
        if m.sum() < min_matches:
            break
        w = (1.0 - dist[m] / max_dist) + 1e-3
        src_m = P_src[m]; dst_m = P_dst[idx[m]]
        mu_s = (src_m * w[:, None]).sum(0) / w.sum()
        mu_d = (dst_m * w[:, None]).sum(0) / w.sum()
        s_c = src_m - mu_s; d_c = dst_m - mu_d
        W = w[:, None]
        var_s = (W * s_c ** 2).sum()
        var_d = (W * d_c ** 2).sum()
        s_new = np.sqrt(var_d / max(var_s, 1e-12))
        H = (s_c * w[:, None]).T @ d_c
        U, _, Vt = np.linalg.svd(H)
        d_ = np.sign(np.linalg.det(Vt.T @ U.T))
        R_new = Vt.T @ np.diag([1, 1, d_]) @ U.T
        t_new = mu_d - s_new * R_new @ mu_s
        s, R, t = s_new, R_new, t_new
        if verbose:
            print(f'[icp] iter {it}: matches={m.sum()}  mean_dist={dist[m].mean():.4f}  s={s:.4f}')
    return s, R, t


def transform_cloud(points_3d, colors, conf, s, R, t, mask_thresh=1e-6):
    m = (np.abs(points_3d[..., 2]) > mask_thresh) & np.isfinite(points_3d[..., 2])
    if not np.any(m):
        return (np.empty((0, 3), np.float32),
                np.empty((0, 3), np.uint8),
                np.empty((0,), np.float32))
    pts = points_3d[m]
    pts_t = (s * (pts @ R.T) + t).astype(np.float32)
    return pts_t, colors[m].astype(np.uint8), conf[m].astype(np.float32)


def gap_fill_consistent(P_ref, C_ref, F_ref, P_next, C_next, F_next,
                        keep_ratio=1.5, verbose=True):
    """Оставляем из next только точки, которые не дублируют ref (там где next
    видит что-то новое, а не то же самое на близкой глубине)."""
    if len(P_ref) == 0:
        return P_next, C_next, F_next
    if len(P_next) == 0:
        return P_ref, C_ref, F_ref

    tree_ref = cKDTree(P_ref)
    d_next_to_ref, _ = tree_ref.query(P_next)

    lower = d_next_to_ref[d_next_to_ref < np.percentile(d_next_to_ref, 90)]
    offset = float(np.median(lower)) if len(lower) > 0 else 0.0
    threshold = keep_ratio * offset

    keep = d_next_to_ref > threshold
    if verbose:
        print(f'[gap-fill] offset={offset:.4f}  threshold={threshold:.4f}  '
              f'keep {keep.sum()} / {len(P_next)}')
    return (np.concatenate([P_ref, P_next[keep]], axis=0),
            np.concatenate([C_ref, C_next[keep]], axis=0),
            np.concatenate([F_ref, F_next[keep]], axis=0))


def voxel_downsample(points, colors, conf, voxel_size=0.01):
    if len(points) == 0:
        return points, colors, conf
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.lexsort((np.abs(points[:, 2]), inv))
    _, first = np.unique(inv[order], return_index=True)
    keep = order[first]
    return points[keep], colors[keep], conf[keep]


def fuse_pairs(pair_ab, pair_bc, use_icp=True, verbose=True):
    """Сшивает cloud_bc в СК cloud_ab.
    1. Аналитическое преобразование по позам камер (точно, не зависит от якорей).
    2. Лёгкий ICP для подчистки остаточной scale-ошибки.
    3. Gap-fill: из bc оставляем только реально новое.
    """
    s, R, t = analytical_transform(pair_ab, pair_bc, verbose=verbose)

    P1, C1, F1 = transform_cloud(pair_ab['points_3d'], pair_ab['colors'],
                                 pair_ab['conf'], 1.0, np.eye(3), np.zeros(3))
    P2, C2, F2 = transform_cloud(pair_bc['points_3d'], pair_bc['colors'],
                                 pair_bc['conf'], 1.0, np.eye(3), np.zeros(3))

    # проверка аналитики на общих якорях
    idx = pair_bc['prev_idx']
    shared = idx >= 0
    if shared.sum() >= 6:
        src_3d = pair_bc['matches'][shared, 4:]
        dst_3d = pair_ab['matches'][idx[shared], 4:]
        pred = s * (src_3d @ R.T) + t
        res = np.linalg.norm(pred - dst_3d, axis=1)
        if verbose:
            print(f'[fuse] analytical residual on {shared.sum()} anchors: '
                  f'mean={res.mean():.4f}  median={np.median(res):.4f}  max={res.max():.4f}')

    if use_icp:
        if verbose:
            print('[fuse] ICP refinement...')
        s, R, t = icp_refine(P2, P1, s, R, t, verbose=verbose)

    P2, C2, F2 = transform_cloud(pair_bc['points_3d'], pair_bc['colors'],
                                 pair_bc['conf'], s, R, t)

    if verbose:
        dm, dmed = cloud_distance(P1, P2, n_sample=5000)
        print(f'[fuse] cloud distance after align: mean={dm:.4f} median={dmed:.4f}')
        save_ply_fast_numpy('cloud_ref.ply',  P1, C1)
        save_ply_fast_numpy('cloud_next.ply', P2, C2)

    P, C, F = gap_fill_consistent(P1, C1, F1, P2, C2, F2, verbose=verbose)
    return P, C, F


# TODO we do not consider vertical stereopairs
def process_stereo_pair(frame_a, frame_b, pts_ab_xyz=None, verbose=True):
    K = get_matrix_K_from_frame(frame_a)

    while True:
        matched_pts_a, matched_pts_b, prev_idx = get_matches_using_optical_flow(
            frame_a, frame_b, prev_points=pts_ab_xyz if pts_ab_xyz is not None else None)

        E, mask = cv2.findEssentialMat(
            matched_pts_a, matched_pts_b, cameraMatrix=K,
            method=cv2.RANSAC, prob=0.999, threshold=0.5)

        points, R, t, mask_pose = cv2.recoverPose(
            E, matched_pts_a, matched_pts_b, cameraMatrix=K, mask=mask)

        if t[0, 0] < 0:
            if verbose:
                plot_flow_vectors(frame_a, matched_pts_a, matched_pts_b, max_arrows=700)
                plot_matches(frame_a, frame_b, matched_pts_a, matched_pts_b)
            break
        else:
            assert False, 'Temporary assert reorder frames'

    valid_mask    = mask.ravel() == 1
    pts_a_inliers = matched_pts_a[valid_mask]
    pts_b_inliers = matched_pts_b[valid_mask]
    prev_idx      = prev_idx[valid_mask]
    inherited     = prev_idx >= 0

    if verbose:
        print(f'[pair] matched={len(matched_pts_a)}  inliers={valid_mask.sum()}  '
              f'inherited_among_inliers={inherited.sum()}')

    if pts_ab_xyz is not None and inherited.sum() >= 3:
        prev_3d = pts_ab_xyz[prev_idx[inherited], 4:]
        P1_tri = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2_tri = K @ np.hstack([R, t.reshape(3, 1)])
        pts4d = cv2.triangulatePoints(
            P1_tri, P2_tri, pts_a_inliers[inherited].T, pts_b_inliers[inherited].T)
        curr_3d = (pts4d[:3] / pts4d[3]).T

        z_prev = np.median(np.abs(prev_3d[:, 2]))
        z_curr = np.median(np.abs(curr_3d[:, 2]))
        if z_curr > 1e-9:
            scale = z_prev / z_curr
            t = t * scale
            if verbose:
                print(f'[pair] z_prev={z_prev:.4f} z_curr={z_curr:.4f} scale={scale:.4f}')

    if verbose:
        plot_flow_vectors(frame_a, pts_a_inliers, pts_b_inliers, max_arrows=700)
        plot_matches(frame_a, frame_b, pts_a_inliers, pts_b_inliers)
        visualize_scene_matplotlib(pts_a_inliers, pts_b_inliers, K, R, t)

    height, width = frame_a.shape[:2]
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K, distCoeffs1=dist_coeffs,
        cameraMatrix2=K, distCoeffs2=dist_coeffs,
        imageSize=(width, height), R=R, T=t.reshape(3, 1))
    assert min(roi1) >= 0

    map1_x, map1_y = cv2.initUndistortRectifyMap(K, None, R1, P1, (width, height), cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(K, None, R2, P2, (width, height), cv2.CV_32FC1)

    rectified_frame_a = cv2.remap(frame_a, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_frame_b = cv2.remap(frame_b, map2_x, map2_y, cv2.INTER_LINEAR)

    if verbose:
        imshow('Rectified frames', np.concatenate([rectified_frame_a, rectified_frame_b], axis=1))

    disp = HitNetDisparity()(rectified_frame_a, rectified_frame_b)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
    disp_visual = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imshow("Disparity Map", disp_visual)

    points_3d = cv2.reprojectImageTo3D(disp, Q)
    conf = np.ones_like(disp)

    percentile = 4
    lo_bound = np.percentile(points_3d[..., 2], percentile)
    up_bound = np.percentile(points_3d[..., 2], 100 - percentile)

    colors = cv2.cvtColor(rectified_frame_a, cv2.COLOR_BGR2RGB)
    mask = (disp > 0.02) & (disp < 120) & np.isfinite(points_3d[..., 2]) & \
           (points_3d[..., 2] < up_bound) & (points_3d[..., 2] > lo_bound)
    roi_mask = np.zeros_like(mask)
    roi_mask[roi1[1]:roi1[1] + roi1[3], roi1[0]:roi1[0] + roi1[2]] = True
    mask = mask & roi_mask

    points_3d[~mask] = 0
    colors[~mask]    = 0
    conf[~mask]      = 0

    pts_a_rectified = cv2.undistortPoints(
        pts_a_inliers.reshape(-1, 1, 2), K, None, R=R1, P=P1).reshape(-1, 2)
    xy = pts_a_rectified.round().astype(int)
    roi_pts_mask = (xy[:, 0] > roi1[0]) & (xy[:, 0] < roi1[0] + roi1[2]) & \
                   (xy[:, 1] > roi1[1]) & (xy[:, 1] < roi1[1] + roi1[3])
    xy = xy[roi_pts_mask]
    pts_a_3d      = points_3d[xy[:, 1], xy[:, 0]]
    pts_a_inliers = pts_a_inliers[roi_pts_mask]
    pts_b_inliers = pts_b_inliers[roi_pts_mask]
    prev_idx      = prev_idx[roi_pts_mask]

    valid_3d = np.abs(pts_a_3d[:, 2]) > 1e-6
    pts_a_3d      = pts_a_3d[valid_3d]
    pts_a_inliers = pts_a_inliers[valid_3d]
    pts_b_inliers = pts_b_inliers[valid_3d]
    prev_idx      = prev_idx[valid_3d]

    if verbose:
        print(f'[pair] pts_a_3d valid={len(pts_a_3d)}  dropped_as_zero={(~valid_3d).sum()}')

    pts_ab_xyz = np.concatenate((pts_a_inliers, pts_b_inliers, pts_a_3d), axis=1)

    if verbose:
        m = points_3d[..., 2] > 0
        save_ply_fast_numpy('myscene.ply', points_3d[m], colors[m])

    return {
        'points_3d': points_3d,
        'colors':    colors,
        'conf':      conf,
        'matches':   pts_ab_xyz,
        'prev_idx':  prev_idx,
        'R': R, 't': t, 'K': K, 'R1': R1,
    }


if __name__ == '__main__':
    video_path = 'sample/kitchen.mp4'
    frames = load_frames(video_path, quiet=True)
    frames = frames[::10]

    frame_a = get_center_crop_coords(frames[2])
    frame_b = get_center_crop_coords(frames[1])
    frame_c = get_center_crop_coords(frames[0])

    print('=== pair_ab ===')
    pair_ab = process_stereo_pair(frame_a, frame_b)

    print('=== pair_bc ===')
    pair_bc = process_stereo_pair(frame_b, frame_c, pts_ab_xyz=pair_ab['matches'])

    print('=== fuse ===')
    P, C, F = fuse_pairs(pair_ab, pair_bc, use_icp=True, verbose=True)

    print('=== voxel ===')
    extent = np.linalg.norm(P.max(0) - P.min(0))
    voxel_size = extent / 400.0
    print(f'before: {len(P)} points, extent={extent:.4f}, voxel_size={voxel_size:.4f}')
    P, C, F = voxel_downsample(P, C, F, voxel_size=voxel_size)
    print(f'after:  {len(P)} points')

    save_ply_fast_numpy('fused.ply', P, C)
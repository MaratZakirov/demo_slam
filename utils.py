import cv2
import numpy as np
import onnxruntime as ort
import os
from imu import *

def load_frames(video_path: str, scale_factor=0.3, quiet=True) -> list:
    cap = cv2.VideoCapture(video_path)

    frame_cnt = 0

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео по пути {video_path}")
        return []

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    frames = []

    while True:
        ret, frame = cap.read()
        frame_cnt += 1

        # Если кадры закончились или ошибка чтения — выходим
        if not ret:
            break

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if not quiet:
            cv2.imshow("Resized Gray Video", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Обработка прервана пользователем.")
            break

        frames.append(resized_frame)

    return frames

def undistort_frames(frames: list, dist: np.ndarray, K: np.ndarray) -> list:
    h, w = frames[0].shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    return [cv2.remap(f, map1, map2, interpolation=cv2.INTER_LINEAR) for f in frames]

def get_matches_using_optical_flow(frame_left, frame_right, pts_i_inherited=np.zeros((0, 2)), max_features=1000, H=None):
    # Находит сопоставленные точки между двумя кадрами с использованием
    # локального оптического потока Лукаса-Канаде
    # For initial setup may use Homography matrix
    #
    # Returns:
    #    pts_i   (np.ndarray): Координаты точек на первом кадре (K x 2)
    #    pts_ip1 (np.ndarray): Координаты этих же точек на втором кадре (K x 2)
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # 2. Находим надежные углы ТОЛЬКО на первом кадре
    pts_i_raw = cv2.goodFeaturesToTrack(
        gray_left,
        maxCorners=max_features,
        qualityLevel=0.01,  # Порог качества (чем выше, тем строже отбор углов)
        minDistance=10  # Минимальное расстояние в пикселях между точками
    )

    if pts_i_raw is None:
        pts_i_raw = np.empty((0, 1, 2), dtype=np.float32)

    N = len(pts_i_raw)
    pts_i_raw = np.concatenate([pts_i_raw, pts_i_inherited.reshape(-1, 1, 2)], dtype=np.float32, axis=0)

    if H is not None:
        # Here we calculate Homography matrix and use for enhance candidates for LK
        xy = pts_i_raw.reshape(-1, 2).astype(np.float64)
        homog = np.hstack([xy, np.ones((len(xy), 1))])
        pred = (H @ homog.T).T
        pred = pred[:, :2] / pred[:, 2:3]
        pts_seed = pred.reshape(-1, 1, 2).astype(np.float32)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    else:
        pts_seed = None
        flags = 0

    # Настраиваем параметры локального поиска Лукаса-Канаде
    lk_params = dict(
        winSize=(21, 21),  # Ищем точку строго в локальном окне 21x21 пиксель
        maxLevel=3,  # Пирамиды изображений для отслеживания быстрых сдвигов
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=flags
    )

    # Вычисляем оптический поток (смещение точек на кадр B)
    # status == 1 для точек, которые успешно нашлись в локальном окне на кадре B
    pts_ip1_raw, status, err = cv2.calcOpticalFlowPyrLK(gray_left, gray_right, pts_i_raw, pts_seed, **lk_params)

    # Фильтруем точки с помощью NumPy-маски status
    valid_mask = (status == 1).reshape(-1)
    pts_i = pts_i_raw[:N][valid_mask[:N]].reshape(-1, 2)
    pts_ip1 = pts_ip1_raw[:N][valid_mask[:N]].reshape(-1, 2)

    pts_ip1_inhereted = pts_ip1_raw[N:].reshape(-1, 2)
    assert len(pts_ip1_inhereted) == len(pts_i_inherited)

    # return artifacts plus residual mask of validness for previois matches
    return pts_i, pts_ip1, pts_ip1_inhereted, valid_mask[N:]

def select_key_frames(frames: list, drop: int, min_survival: float = 0.7):
    # Select heuristically frames which are far enough in optical flow sense
    # to have good base but not too far enough to still be able to track using optical flow
    frames = frames[::drop]
    N = len(frames)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    key_idx = []
    last_kf = 0
    while last_kf < N:
        pts_k = cv2.goodFeaturesToTrack(gray[last_kf], maxCorners=1000,
                                        qualityLevel=0.01, minDistance=10)
        if pts_k is None:
            last_kf += 1
            continue

        key_idx.append(last_kf)

        for i in range(last_kf + 1, N):
            _, status, _ = cv2.calcOpticalFlowPyrLK(gray[last_kf], gray[i], pts_k, None, **lk_params)
            if float(status.mean()) < min_survival:
                last_kf = i
                break
        else:
            break

    return [frames[j] for j in key_idx], np.array(key_idx)

def decompose_2d_vector_field(xy: np.ndarray, uv: np.ndarray):
    xy_centered = xy - np.mean(xy, axis=0)
    uv_centered = uv - np.mean(uv, axis=0)
    J_T, _, _, _ = np.linalg.lstsq(xy_centered, uv_centered, rcond=None)
    J = J_T.T  # Транспонируем, чтобы получить стандартный Якобиан 2х2
    du_dx = J[0, 0]
    du_dy = J[0, 1]
    dv_dx = J[1, 0]
    dv_dy = J[1, 1]
    rotation = 0.5 * (dv_dx - du_dy)  # Величина глобального вращения
    shear = 0.5 * (du_dy + dv_dx)     # Величина чистая сдвига
    return rotation, shear

def get_matrix_K_from_frame(frame):
    height, width = frame.shape[:2]

    fx = width
    fy = width

    cx = width / 2.0
    cy = height / 2.0

    # 4. Собираем матрицу K размером (3, 3) типа float32
    K = np.array([
                   [fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]
    ], dtype=np.float32)

    return K

def apply_roi(rect_a, rect_b, roi_a, roi_b):
    x1, y1, w1, h1 = roi_a
    x2, y2, w2, h2 = roi_b

    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    h = min(y1 + h1, y2 + h2) - y

    rect_a = rect_a[y:y + h, x:x + w]
    rect_b = rect_b[y:y + h, x:x + w]

    return rect_a, rect_b

def get_center_crop_coords(frame, target_w=720, target_h=1280):
    img_h, img_w = frame.shape[:2]

    # Вычисляем целевые пропорции (для 720x1280 это 0.5625)
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h

    # Определяем размеры рамки кропа в пикселях оригинала
    if img_aspect < target_aspect:
        # Кадр слишком широкий — берем всю высоту, сужаем ширину
        crop_h = img_h
        crop_w = int(img_h * target_aspect)
    else:
        # Кадр слишком узкий/высокий — берем всю ширину, урезаем высоту
        crop_w = img_w
        crop_h = int(img_w / target_aspect)

    # Находим координаты центра
    x_offset = (img_w - crop_w) // 2
    y_offset = (img_h - crop_h) // 2

    # Возвращаем кадрированный фрейм в исходном разрешении
    frame = frame[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]

    aspect_source = frame.shape[0] / frame.shape[1]
    aspect_target = target_w / target_h

    np.isclose(aspect_target, aspect_source, atol=1e-5, rtol=1e-5)

    return frame

# Calculate relative pose from global
def relative_pose(pose_i, pose_j):
    R_i, t_i = pose_i
    R_j, t_j = pose_j
    R_ji = R_j.T @ R_i
    t_ji = R_j.T @ (np.asarray(t_i).ravel() - np.asarray(t_j).ravel())
    return R_ji, t_ji.reshape(3, 1)

class HitNetDisparity:
    def __init__(self, model_path="models/model_float32.onnx", height=720, width=1280):
        self.model_path = model_path
        self.height = height
        self.width = width
        self.input_name = "input"

        onnx_opts = ort.SessionOptions()
        onnx_opts.intra_op_num_threads = 6
        onnx_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        onnx_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, onnx_opts)

    def __call__(self, rect_a, rect_b):
        resized_a = cv2.resize(rect_a, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resized_b = cv2.resize(rect_b, (self.width, self.height), interpolation=cv2.INTER_AREA)

        rgb_a = cv2.cvtColor(resized_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_b = cv2.cvtColor(resized_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        outputs = self.session.run(None, {
            self.input_name: np.concatenate((
                np.transpose(rgb_a, (2, 0, 1)),
                np.transpose(rgb_b, (2, 0, 1))
            ), axis=0)[None]
        })

        disp_float = np.squeeze(outputs[0])

        orig_h, orig_w = rect_a.shape[:2]
        disp_resized = cv2.resize(disp_float, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        scale_factor = orig_w / self.width
        disp_resized = disp_resized * scale_factor

        disp_resized[disp_resized <= 0.1] = np.nan

        return disp_resized

def voxel_vote(all_pts, all_cols, all_i, voxel_size=0.1,
               min_count=3, min_pairs=2):
    # Voting voxels.
    # - rounding coordinates to voxel_size
    # - get mean in all voxels for XYZ and RGB
    # - keep where count >= min_count and number of pairs >= min_pairs

    # voxel grid ---
    keys = np.floor(all_pts / voxel_size).astype(np.int64)   # (N, 3)

    # hashing and reverse indexing
    uniq_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    n_vox = len(uniq_keys)

    # voxel counters ---
    counts = np.bincount(inv, minlength=n_vox)

    sum_xyz = np.stack([
        np.bincount(inv, weights=all_pts[:, d], minlength=n_vox)
        for d in range(3)
    ], axis=1)                                                # (n_vox, 3)

    cols_f = all_cols.astype(np.float64)
    sum_rgb = np.stack([
        np.bincount(inv, weights=cols_f[:, d], minlength=n_vox)
        for d in range(3)
    ], axis=1)

    # mean values
    mean_xyz = sum_xyz / counts[:, None]
    mean_rgb = (sum_rgb / counts[:, None]).astype(np.uint8)

    # number of unique pairs
    # coding (inv, pair) into one unique int64
    combo = inv.astype(np.int64) * (all_i.max() + 1) + all_i.astype(np.int64)
    uniq_combo = np.unique(combo)
    vox_of_combo = uniq_combo // (all_i.max() + 1)
    pair_counts = np.bincount(vox_of_combo, minlength=n_vox)

    # Filtering
    keep = (counts >= min_count) & (pair_counts >= min_pairs)
    print(f"[voxel] in={len(all_pts)}  voxels={n_vox}  "
          f"kept={keep.sum()}  ({keep.mean()*100:.1f}%)  "
          f"counts: med={int(np.median(counts[keep])) if keep.any() else 0}  "
          f"pairs: med={int(np.median(pair_counts[keep])) if keep.any() else 0}")

    return mean_xyz[keep], mean_rgb[keep]
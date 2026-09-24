import cv2
import numpy as np
import onnxruntime as ort
import os

def load_frames(video_path: str, quiet=True) -> list:
    cap = cv2.VideoCapture(video_path)

    frame_cnt = 0

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео по пути {video_path}")
        return []

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale_factor = 0.3
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

def get_matches_using_optical_flow(frame_a, frame_b, max_features=1000):
    """
    Находит сопоставленные точки между двумя кадрами с использованием
    локального оптического потока Лукаса-Канаде вместо глобального SIFT.

    Parameters:
        frame_a (np.ndarray): Первый цветной кадр (H x W x 3)
        frame_b (np.ndarray): Второй цветной кадр (H x W x 3)
        max_features (int): Максимальное число углов для поиска на первом кадре

    Returns:
        matched_pts_a (np.ndarray): Координаты точек на первом кадре (K x 2)
        matched_pts_b (np.ndarray): Координаты этих же точек на втором кадре (K x 2)
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # 2. Находим надежные углы ТОЛЬКО на первом кадре
    pts_a_raw = cv2.goodFeaturesToTrack(
        gray_a,
        maxCorners=max_features,
        qualityLevel=0.01,  # Порог качества (чем выше, тем строже отбор углов)
        minDistance=10  # Минимальное расстояние в пикселях между точками
    )

    # Если на первом кадре вообще нет стабильных точек, возвращаем пустые массивы
    if pts_a_raw is None or len(pts_a_raw) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # 3. Настраиваем параметры локального поиска Лукаса-Канаде
    lk_params = dict(
        winSize=(21, 21),  # Ищем точку строго в локальном окне 21x21 пиксель
        maxLevel=3,  # Пирамиды изображений для отслеживания быстрых сдвигов
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # 4. Вычисляем оптический поток (смещение точек на кадр B)
    # status == 1 для точек, которые успешно нашлись в локальном окне на кадре B
    pts_b_raw, status, err = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, pts_a_raw, None, **lk_params)

    # 5. Фильтруем точки с помощью NumPy-маски status
    valid_mask = (status == 1).reshape(-1)

    # Избавляемся от лишней оси координат OpenCV (N, 1, 2) -> (K, 2)
    matched_pts_a = pts_a_raw[valid_mask].reshape(-1, 2)
    matched_pts_b = pts_b_raw[valid_mask].reshape(-1, 2)

    return matched_pts_a, matched_pts_b

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
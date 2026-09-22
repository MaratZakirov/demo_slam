import cv2
import numpy as np

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

def triangulate_points_numpy(pts_a, pts_b, K, R, t):
    """
    Триангуляция точек на чистом NumPy с использованием SVD.

    Parameters:
        pts_a (np.ndarray): Точки с первого кадра (N x 2)
        pts_b (np.ndarray): Точки со второго кадра (N x 2)
        K (np.ndarray): Матрица камеры (3 x 3)
        R (np.ndarray): Матрица вращения (3 x 3)
        t (np.ndarray): Вектор сдвига (3 x 1)

    Returns:
        points_3d (np.ndarray): Массив 3D координат (N x 3)
    """
    # 1. Строим матрицы проекции для обеих камер (размер 3x4)
    # Первая камера в начале координат
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # Вторая камера смещена на R и t
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    num_points = pts_a.shape[0]
    points_3d = []

    # Решаем систему уравнений для каждой пары точек
    for i in range(num_points):
        u1, v1 = pts_a[i]
        u2, v2 = pts_b[i]

        # Составляем матрицу А (размер 4x4) на основе уравнений проекции
        A = np.zeros((4, 4))
        A[0] = u1 * P1[2, :] - P1[0, :]
        A[1] = v1 * P1[2, :] - P1[1, :]
        A[2] = u2 * P2[2, :] - P2[0, :]
        A[3] = v2 * P2[2, :] - P2[1, :]

        # Решаем систему A * X = 0 через SVD разложение в NumPy
        _, _, Vt = np.linalg.svd(A)

        # Решение — это последняя строка матрицы Vt (вектор с наименьшим сингулярным числом)
        X_homogeneous = Vt[-1]

        # Переводим из однородных координат в обычные 3D (делением на четвертую компоненту)
        X_3d = X_homogeneous[:3] / X_homogeneous[3]
        points_3d.append(X_3d)

    return np.array(points_3d)

def compute_disparity_subpixel_numpy(rect_a, rect_b, window_size=15, max_disp=80):
    """
    Расчет ПЛАВНОЙ карты диспаратности на чистом NumPy с субпиксельной интерполяцией.
    """
    gray_a = cv2.cvtColor(rect_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(rect_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray_a.shape
    half_w = window_size // 2

    # Создаем 3D-куб NumPy размера (max_disp, H, W), чтобы сохранить значения ошибок для каждого сдвига
    # Это необходимо для последующего поиска соседей (d-1) и (d+1)
    sad_cube = np.full((max_disp, h, w), np.inf, dtype=np.float32)

    min_sad = np.full((h, w), np.inf, dtype=np.float32)
    disparity_map_int = np.zeros((h, w), dtype=np.int32)

    kernel = np.ones((window_size, window_size), dtype=np.float32)

    # 1. Сбор данных (тот же цикл, но сохраняем историю SAD)
    for d in range(max_disp):
        if d >= w:
            break

        shifted_b = np.zeros_like(gray_b)
        shifted_b[:, :w - d] = gray_b[:, d:]

        abs_diff = np.abs(gray_a - shifted_b)
        sad = cv2.filter2D(abs_diff, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Небольшое пространственное сглаживание самой стоимости (убирает прыжки пикселей)
        sad = cv2.GaussianBlur(sad, (3, 3), 0)

        valid_zone = np.zeros((h, w), dtype=bool)
        valid_zone[:, d + half_w: w - half_w] = True

        sad_cube[d] = np.where(valid_zone, sad, np.inf)

        better_match = (sad < min_sad) & valid_zone
        min_sad[better_match] = sad[better_match]
        disparity_map_int[better_match] = d

    # 2. МАГИЯ NUMPY: Векторная субпиксельная интерполяция параболой
    # Создаем итоговую float32 карту
    disparity_map_smooth = disparity_map_int.astype(np.float32)

    # Нам нужны только те пиксели, у которых лучший сдвиг лежит внутри диапазона (есть соседи слева и справа)
    # Иначе мы не сможем построить параболу
    subpixel_mask = (disparity_map_int > 0) & (disparity_map_int < max_disp - 1)

    if np.any(subpixel_mask):
        # Вытаскиваем координаты пикселей, проходящих по маске
        y_indices, x_indices = np.where(subpixel_mask)
        d_best = disparity_map_int[subpixel_mask]

        # Получаем значения ошибок SAD для лучшего сдвига, а также для его левого и правого соседей
        # Используем продвинутую индексацию NumPy для мгновенной выборки из 3D-куба
        sad_center = sad_cube[d_best, y_indices, x_indices]
        sad_minus = sad_cube[d_best - 1, y_indices, x_indices]
        sad_plus = sad_cube[d_best + 1, y_indices, x_indices]

        # Математическая формула вершины параболы (коэффициент субпиксельного сдвига)
        # Знаменатель защищаем от деления на ноль микро-числом 1e-5
        denominator = 2.0 * (sad_minus + sad_plus - 2.0 * sad_center)
        subpixel_delta = (sad_minus - sad_plus) / (denominator + 1e-5)

        # Ограничиваем поправку диапазоном [-0.5, 0.5], чтобы математика не улетала из-за шума
        subpixel_delta = np.clip(subpixel_delta, -0.5, 0.5)

        # Записываем дробную поправку обратно в карту диспаратности
        disparity_map_smooth[subpixel_mask] += subpixel_delta

    # ФИНАЛЬНОЕ СГЛАЖИВАНИЕ ВОЛН:
    # Фильтр Гаусса размера 3х3 или 5x5 с микро-радиусом размоет циклическую рябь параболы,
    # превратив волны в идеально ровную, скользящую поверхность.
    disparity_map_final = cv2.GaussianBlur(disparity_map_smooth, (5, 5), 0.5)

    return disparity_map_final

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
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


def extract_sift_features(frame, max_features=1000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Инициализируем и запускаем SIFT
    sift = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Если точек не найдено, возвращаем пустые массивы нужной формы
    if keypoints is None or len(keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    # 3. Вытаскиваем координаты (u, v) из объектов OpenCV в чистый массив NumPy
    points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return points_2d, descriptors


def match_features_numpy(desc_a, desc_b, ratio_threshold=0.75):
    """
    Находит индексы совпадающих точек между двумя кадрами на чистом NumPy.
    """
    if desc_a.shape[0] == 0 or desc_b.shape[0] == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    # 1. Считаем матрицу квадратов Евклидовых расстояний между всеми парами векторов
    # Формула: (a - b)^2 = a^2 + b^2 - 2ab
    a2 = np.sum(desc_a**2, axis=1, keepdims=True)  # (N, 1)
    b2 = np.sum(desc_b**2, axis=1)                 # (M,)
    ab = np.dot(desc_a, desc_b.T)                  # (N, M)
    dists = np.sqrt(np.maximum(a2 + b2 - 2 * ab, 0)) # Матрица расстояний (N, M)

    # 2. Для каждой точки из А находим два ближайших соседа из B
    idx_sorted = np.argsort(dists, axis=1)
    idx_closest1 = idx_sorted[:, 0]  # Индекс лучшего соседа
    idx_closest2 = idx_sorted[:, 1]  # Индекс второго по близости соседа

    # Получаем сами расстояния до них
    d1 = dists[np.arange(len(dists)), idx_closest1]
    d2 = dists[np.arange(len(dists)), idx_closest2]

    # 3. Тест Лоу: первый сосед должен быть значительно ближе второго
    mask = d1 < ratio_threshold * d2

    # Итоговые индексы совпадений
    matches_idx_a = np.where(mask)[0]
    matches_idx_b = idx_closest1[mask]

    return matches_idx_a, matches_idx_b


def get_matches_using_sift(frame_a, frame_b):
    points_2d_a, descriptors_a = extract_sift_features(frame_a)
    points_2d_b, descriptors_b = extract_sift_features(frame_b)

    idx_a, idx_b = match_features_numpy(points_2d_a, points_2d_b)

    matched_pts_a = points_2d_a[idx_a]  # Массив (K, 2)
    matched_pts_b = points_2d_b[idx_b]  # Массив (K, 2)

    return matched_pts_a, matched_pts_b


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
    # 1. Переводим оба кадра в черно-белый формат (обязательно для трекинга)
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


import numpy as np


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


import numpy as np
import cv2


def compute_disparity_numpy(rect_a, rect_b, window_size=7, max_disp=64):
    """
    Расчет карты диспаратности на чистом NumPy методом SAD.
    """
    # 1. Переводим в gray float32 для точности математических операций
    gray_a = cv2.cvtColor(rect_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(rect_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray_a.shape
    half_w = window_size // 2

    # Сюда будем складывать минимальную стоимость (ошибку) для каждого пикселя
    min_sad = np.full((h, w), np.inf, dtype=np.float32)
    # Сюда запишем итоговый сдвиг (диспаратность)
    disparity_map = np.zeros((h, w), dtype=np.float32)

    # Интегральное окно (фильтр размытия) заменяет нам циклы по пикселям окна
    kernel = np.ones((window_size, window_size), dtype=np.float32)

    # Основной цикл по всем возможным сдвигам (поиск по эпиполярной линии)
    for d in range(max_disp):
        if d >= w:
            break

        # Сдвигаем правый кадр влево на 'd' пикселей с помощью NumPy
        # Заполняем освободившиеся пиксели справа нулями
        shifted_b = np.zeros_like(gray_b)
        shifted_b[:, :w - d] = gray_b[:, d:]

        # Считаем абсолютную разность между левым кадром и сдвинутым правым
        abs_diff = np.abs(gray_a - shifted_b)

        # Суммируем разности в пределах окна размера window_size
        # cv2.filter2D здесь используется как супер-быстрая замена np.convolve2d
        sad = cv2.filter2D(abs_diff, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Маска: где текущая ошибка меньше, чем сохраненная ранее
        # Важно искать только там, где пиксель физически существует (w > d)
        valid_zone = np.zeros((h, w), dtype=bool)
        valid_zone[:, d + half_w:w - half_w] = True

        better_match = (sad < min_sad) & valid_zone

        # Обновляем минимальную ошибку и запоминаем этот сдвиг 'd'
        min_sad[better_match] = sad[better_match]
        disparity_map[better_match] = d

    return disparity_map



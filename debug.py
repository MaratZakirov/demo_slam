import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def imshow(label: str, image: np.ndarray, use_cv=False):
    if use_cv:
        cv2.imshow(label, image)
        cv2.waitKey(0)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(label, fontsize=14, fontweight='bold')
        plt.show()

def plot_flow_vectors(frame_a, points_2d_a, points_2d_b, max_arrows=100,
                      scale=1.0, min_len=0.5, draw_points=True):
    """
    Рисует вектора оптического потока на первом кадре.

    Parameters:
        frame_a (np.ndarray): Первый цветной кадр
        points_2d_a (np.ndarray): Точки на первом кадре (N x 2), порядок (x, y)
        points_2d_b (np.ndarray): Точки на втором кадре (N x 2), порядок (x, y)
        max_arrows (int): Максимум стрелок для отрисовки
        scale (float): Множитель длины стрелки (для визуального усиления)
        min_len (float): Минимальная длина вектора, чтобы его рисовать (в пикселях)
        draw_points (bool): Рисовать ли сами точки
    """
    canvas = frame_a.copy()

    # Вектора смещения: (dx, dy)
    flow = points_2d_b - points_2d_a
    lengths = np.linalg.norm(flow, axis=1)

    # Фильтруем слишком короткие вектора — они визуально мусор
    mask = lengths >= min_len
    idxs = np.where(mask)[0]

    # Ограничиваем количество
    if len(idxs) > max_arrows:
        # Берём самые длинные — они нагляднее
        idxs = idxs[np.argsort(lengths[idxs])[::-1][:max_arrows]]

    # Цвет по направлению угла (HSV → BGR), чтобы было видно куда кто идёт
    for i in idxs:
        pt_a = points_2d_a[i]
        dx, dy = flow[i] * scale

        p1 = (int(round(pt_a[0])), int(round(pt_a[1])))
        p2 = (int(round(pt_a[0] + dx)), int(round(pt_a[1] + dy)))

        # Угол в градусах [0..360) для раскраски
        angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        hue = int(angle / 2.0)  # 0..180 для OpenCV HSV
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        # Стрелка
        cv2.arrowedLine(canvas, p1, p2, color, 2, cv2.LINE_AA, tipLength=0.3)

        if draw_points:
            cv2.circle(canvas, p1, 2, color, -1)

    imshow(f"Optical Flow (arrows={len(idxs)}, scale={scale})", canvas)

def plot_matches(frame_a, frame_b, points_2d_a, points_2d_b, max_lines=30):
    """
    Визуализирует сопоставленные точки, рисуя линии между двумя кадрами.

    Parameters:
        frame_a (np.ndarray): Первый цветной кадр
        frame_b (np.ndarray): Второй цветной кадр
        points_2d_a (np.ndarray): Координаты точек на первом кадре (N x 2)
        points_2d_b (np.ndarray): Координаты точек на втором кадре (N x 2)
        max_lines (int): Максимальное количество линий для отрисовки (чтобы не перегружать картинку)
    """
    # 1. Склеиваем два кадра по горизонтали на чистом NumPy
    # Результат будет иметь ширину (width_a + width_b)
    h_a, w_a = frame_a.shape[:2]
    h_b, w_b = frame_b.shape[:2]

    # Высота должна совпадать. Если нет — подгоняем под первый кадр
    if h_a != h_b:
        frame_b = cv2.resize(frame_b, (w_b, h_a), interpolation=cv2.INTER_AREA)
        w_b = frame_b.shape[1]

    canvas = np.hstack((frame_a, frame_b))

    # Ограничиваем количество линий для наглядности
    num_matches = min(len(points_2d_a), max_lines)

    # Генератор случайных цветов для каждой линии
    np.random.seed(42)  # Чтобы цвета не моргали при каждом запуске одинаково
    colors = np.random.randint(0, 255, size=(num_matches, 3), dtype=np.int32)

    # 2. Рисуем линии поверх склеенного кадра
    for i in range(num_matches):
        # Координаты на первом кадре
        pt_a = tuple(points_2d_a[i].astype(np.int32))

        # Координаты на втором кадре нужно сдвинуть вправо на ширину первого кадра w_a
        pt_b = (int(points_2d_b[i][0] + w_a), int(points_2d_b[i][1]))

        color = [int(c) for c in colors[i]]

        # Рисуем точку на первом кадре, на втором и соединяющую их линию
        cv2.circle(canvas, pt_a, 4, color, -1)
        cv2.circle(canvas, pt_b, 4, color, -1)
        cv2.line(canvas, pt_a, pt_b, color, 1, cv2.LINE_AA)

    # 3. Показываем результат
    imshow(f"Matches Debug (Showing {num_matches}/{len(points_2d_a)} lines)", canvas)

def save_obj_fast_numpy(filename, points, colors):
    """
    Молниеносно сохраняет 3D точки и цвета в простейший формат OBJ
    на чистом NumPy без циклов Python.
    """
    # 1. Нормализуем цвета: в OBJ цвета должны быть от 0.0 до 1.0 (float)
    colors_float = colors.astype(np.float32) / 255.0

    # 2. Склеиваем координаты и цвета по горизонтали в одну гигантскую матрицу (N, 6)
    # [X, Y, Z, R, G, B]
    data = np.hstack((points, colors_float))

    # 3. Сбрасываем всю матрицу в файл за один раз средствами NumPy.
    # fmt='v %.4f %.4f %.4f %.4f %.4f %.4f' означает:
    # в начале строки ставить букву 'v' (vertex), а затем 6 чисел с плавающей точкой
    np.savetxt(
        filename,
        data,
        fmt='v %.4f %.4f %.4f %.4f %.4f %.4f',
        comments=''  # Убираем автоматические решетки '#' от NumPy в начале файла
    )

def save_ply_fast_numpy(filename, points, colors):
    """
    Молниеносно сохраняет 3D облако точек с цветами в текстовый формат PLY.
    """
    # 1. Для PLY цвета должны быть целыми числами от 0 до 255 (uint8)
    colors_uint8 = colors.astype(np.uint8)

    # 2. Склеиваем координаты (float32) и цвета (будут приведены к float)
    data = np.hstack((points.astype(np.float32), colors_uint8.astype(np.float32)))

    # 3. Формируем правильный заголовок PLY для облака точек
    header = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header"

    # 4. Сохраняем одной векторной строкой без циклов
    np.savetxt(
        filename,
        data,
        fmt='%.4f %.4f %.4f %d %d %d',
        header=header,
        comments=''
    )


def visualize_scene_matplotlib(pts_a, pts_b, K, R, t):
    """
    Интерактивная 3D визуализация 35 точек и реальной геометрии двух камер в Matplotlib.
    """
    # --- 1. Триангуляция точек на чистом NumPy (DLT метод) ---
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    cloud_3d = []
    for i in range(len(pts_a)):
        u1, v1 = pts_a[i]
        u2, v2 = pts_b[i]
        A = np.zeros((4, 4))
        A[0] = u1 * P1[2, :] - P1[0, :]
        A[1] = v1 * P1[2, :] - P1[1, :]
        A[2] = u2 * P2[2, :] - P2[0, :]
        A[3] = v2 * P2[2, :] - P2[1, :]
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X_3d = X[:3] / X[3]
        cloud_3d.append(X_3d)
    cloud_3d = np.array(cloud_3d)

    # --- 2. Функция генерации реального каркаса камеры ---
    def get_camera_mesh(R_cam, t_cam, K_matrix, scale=0.001):
        # scale преобразует пиксельные размеры K в удобный масштаб для графика
        fx = K_matrix[0, 0] * scale
        cx = K_matrix[0, 2] * scale
        cy = K_matrix[1, 2] * scale

        # 5 базовых точек пирамиды в локальных координатах
        local_pts = np.array([
            [0, 0, 0],  # Оптический центр (вершина)
            [-cx, -cy, fx],  # Левый-верхний угол матрицы
            [cx, -cy, fx],  # Правый-верхний угол
            [cx, cy, fx],  # Правый-нижний угол
            [-cx, cy, fx]  # Левый-нижний угол
        ])

        # Переводим в мировую систему координат первой камеры
        # X_world = R.T @ (X_local - t)
        t_vec = t_cam.reshape(1, 3)
        world_pts = (local_pts - t_vec) @ R_cam
        return world_pts

    # Генерируем вершины для обеих камер
    cam1_nodes = get_camera_mesh(np.eye(3), np.zeros((3, 1)), K)
    cam2_nodes = get_camera_mesh(R, t, K)

    # Индексы линий, формирующих каркас пирамиды
    cam_lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    # --- 3. Строим интерактивный 3D график Matplotlib ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Рисуем 35 триангулированных 3D-точек (синие сферы)
    ax.scatter(cloud_3d[:, 0], cloud_3d[:, 1], cloud_3d[:, 2], c='blue', s=30, label='3D Points (Inliers)')

    # Отрисовка Первой Камеры (Зеленая пирамида)
    for line in cam_lines:
        p1, p2 = cam1_nodes[line[0]], cam1_nodes[line[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='green', linewidth=2)
    ax.scatter(cam1_nodes[0, 0], cam1_nodes[0, 1], cam1_nodes[0, 2], c='green', s=100, label='Camera 1 (Start)')

    # Отрисовка Второй Камеры (Красная пирамида)
    for line in cam_lines:
        p1, p2 = cam2_nodes[line[0]], cam2_nodes[line[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red', linewidth=2)
    ax.scatter(cam2_nodes[0, 0], cam2_nodes[0, 1], cam2_nodes[0, 2], c='red', s=100, label='Camera 2 (Moved)')

    # Настройка осей координат по стандарту OpenCV (ось Y вниз, Z вперед)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Depth)')
    ax.invert_yaxis()  # Переворачиваем Y, чтобы графический верх был верхом в реальности

    ax.legend()
    plt.title("Отладка геометрии SLAM: Взаимное положение камер и 3D-точек")
    plt.show()

def show_3d_match(pa_3d, pb_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pa_3d[:, 0], pa_3d[:, 1], pa_3d[:, 2], c='blue', s=30, label='3D Points A')
    ax.scatter(pb_3d[:, 0], pb_3d[:, 1], pb_3d[:, 2], c='red',  s=30, label='3D Points B')

    # Настройка осей координат по стандарту OpenCV (ось Y вниз, Z вперед)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Depth)')
    ax.invert_yaxis()  # Переворачиваем Y, чтобы графический верх был верхом в реальности

    ax.legend()
    plt.title("Отладка геометрии SLAM: Взаимное положение камер и 3D-точек")
    plt.show()

def visualize_slam_scene(data, K=None, frustum_frac=0.03, point_size=10,
                         save_path=None, box_aspect=None, debug=False):
    """
    Рисует все камеры и их 3D точки в системе кадра 0.
    Каждая камера и её точки имеют один и тот же цвет.

    Parameters
    ----------
    data : list of [R_i, t_i, pts_2d_i, pts_3d_i]
        R_i, t_i — поза камеры i в системе кадра 0: X_0 = R_i @ X_i + t_i
        pts_3d_i — 3D точки пары i, уже переведённые в систему кадра 0
    K : np.ndarray | None
        Матрица внутренних параметров 3x3. Если задана — фрустум строится
        по реальному полю зрения камеры. Если None — абстрактная пирамида.
    frustum_frac : float
        Глубина фрустума как доля от габарита сцены. Default 0.03 — маленькие
        аккуратные пирамидки, не перекрывающие друг друга.
    point_size : int
        Размер маркера 3D точек.
    save_path : str | None
        Если задан — сохраняет картинку.
    box_aspect : tuple | None
        Явное соотношение сторон осей 3D. Если None — (1, 1, 1).
    debug : bool
        Печатает диагностику по позам камер и габаритам.
    """
    num_cameras = len(data)

    # ---- 0. Диагностика ----
    if debug:
        print("=" * 60)
        print(f"[visualize_slam_scene] cameras={num_cameras}")
        for i, (R_i, t_i, _, pts_3d) in enumerate(data):
            c = np.asarray(t_i).ravel()
            detR = np.linalg.det(R_i)
            ortho = np.linalg.norm(R_i @ R_i.T - np.eye(3))
            pts = np.asarray(pts_3d)
            z_med = np.median(pts[:, 2]) if pts.size else float('nan')
            print(f"  cam {i}: center=[{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}]  "
                  f"det(R)={detR:+.4f}  |RR^T-I|={ortho:.2e}  "
                  f"pts={len(pts)}  z_med={z_med:+.3f}")
        print("=" * 60)

    # ---- 1. Оценка габарита сцены по всем точкам ----
    pts_all = [np.asarray(d[3]) for d in data if len(d[3]) > 0]
    if not pts_all:
        print("[visualize_slam_scene] Нет 3D точек для отображения")
        return
    pts_all = np.vstack(pts_all)

    lo = np.percentile(pts_all, 2, axis=0)
    hi = np.percentile(pts_all, 98, axis=0)
    scene_extent = float(np.linalg.norm(hi - lo))
    if scene_extent < 1e-9:
        scene_extent = 1.0

    if debug:
        print(f"[visualize_slam_scene] scene_extent={scene_extent:.4f}")
        print(f"   lo={lo},  hi={hi}")

    # ---- 2. Шаблон фрустума в ЛОКАЛЬНЫХ координатах камеры ----
    # OpenCV: X вправо, Y вниз, Z вперёд
    d = frustum_frac * scene_extent

    if K is None:
        # Абстрактная симметричная пирамида
        w = 0.7 * d
        h = 0.5 * d
        local_pts = np.array([
            [ 0,  0, 0],
            [-w, -h, d],
            [ w, -h, d],
            [ w,  h, d],
            [-w,  h, d],
        ], dtype=np.float64)
    else:
        # Реальный фрустум по K
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        cx, cy = K[0, 2], K[1, 2]

        # Границы image plane в пикселях (предполагаем cx,cy — центр кадра)
        W = 2.0 * cx
        H = 2.0 * cy

        pixels = np.array([
            [cx, cy],   # 0: центр (луч по Z)
            [0,  0],    # 1: ЛВ
            [W,  0],    # 2: ПВ
            [W,  H],    # 3: ПН
            [0,  H],    # 4: ЛН
        ], dtype=np.float64)

        # K^-1 @ [u, v, 1]^T — нормированный луч. K^-1 сам вычитает cx,cy.
        rays = np.hstack([pixels, np.ones((5, 1))])
        local_pts = (np.linalg.inv(K) @ rays.T).T

        # Оптический центр в ноль
        local_pts[0] = 0.0

        # Углы на глубину d
        local_pts[1:] *= (d / local_pts[1:, 2:3])

        if debug:
            print("[visualize_slam_scene] local frustum points:")
            for j, p in enumerate(local_pts):
                print(f"   {j}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")

    cam_lines = [[0, 1], [0, 2], [0, 3], [0, 4],
                 [1, 2], [2, 3], [3, 4], [4, 1]]

    # ---- 3. Палитра ----
    cmap = plt.get_cmap('tab10', max(num_cameras, 10))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, (R_i, t_i, pts_2d, pts_3d) in enumerate(data):
        color = cmap(i % 10)

        # --- 3D точки камеры i (уже в системе 0) ---
        pts = np.asarray(pts_3d)
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=[color], s=point_size, alpha=0.65,
                       label=f'Cam {i}  ({len(pts)} pts)')

        # --- Фрустум камеры i: X_0 = R_i @ X_local + t_i ---
        t_vec = np.asarray(t_i).reshape(1, 3)
        world_pts = (R_i @ local_pts.T).T + t_vec

        for line in cam_lines:
            p1, p2 = world_pts[line[0]], world_pts[line[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=2, alpha=0.9)

        # --- Центр камеры ---
        c_world = np.asarray(t_i).ravel()
        ax.scatter(c_world[0], c_world[1], c_world[2],
                   c=[color], s=120, marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5)

        # подпись камеры
        ax.text(c_world[0], c_world[1], c_world[2], f' cam{i}',
                color=color, fontsize=10, fontweight='bold')

    # ---- 4. Оси кадра 0 для ориентации ----
    axis_len = 0.15 * scene_extent
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.15)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.15)

    # ---- 5. Оформление ----
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Depth)')
    ax.invert_yaxis()

    if box_aspect is None:
        box_aspect = (1.0, 1.0, 1.0)
    ax.set_box_aspect(box_aspect)

    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=-60)

    ax.legend(loc='upper left', fontsize=8)
    plt.title("SLAM scene in frame 0 coordinate system")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
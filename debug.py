import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    cv2.imshow(f"Matches Debug (Showing {num_matches}/{len(points_2d_a)} lines)", canvas)
    cv2.waitKey(0)  # Ждем нажатия любой клавиши, чтобы продолжить

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
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='green', linewidth=2)
    ax.scatter(cam1_nodes[0, 0], cam1_nodes[0, 1], cam1_nodes[0, 2], c='green', s=100, label='Camera 1 (Start)')

    # Отрисовка Второй Камеры (Красная пирамида)
    for line in cam_lines:
        p1, p2 = cam2_nodes[line[0]], cam2_nodes[line[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='red', linewidth=2)
    ax.scatter(cam2_nodes[0, 0], cam2_nodes[0, 1], cam2_nodes[0, 2], c='red', s=100, label='Camera 2 (Moved)')

    # Настройка осей координат по стандарту OpenCV (ось Y вниз, Z вперед)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Depth)')
    ax.invert_yaxis()  # Переворачиваем Y, чтобы графический верх был верхом в реальности

    ax.legend()
    plt.title("Отладка геометрии SLAM: Взаимное положение камер и 3D-точек")
    plt.show()

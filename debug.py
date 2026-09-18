import cv2
import numpy as np


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



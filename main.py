from utils import *
from debug import *
import matplotlib.pyplot as plt

video_path = 'sample/mnist.mp4'

frames = load_frames(video_path, quiet=True)

frames = frames[::10]

K = get_matrix_K_from_frame(frames[0])

#matched_pts_a, matched_pts_b = get_matches_using_sift(frames[0], frames[1])
matched_pts_a, matched_pts_b = get_matches_using_optical_flow(frames[0], frames[1])

plot_matches(frames[0], frames[1], matched_pts_a, matched_pts_b)

# К — матрица внутренних параметров вашей камеры (Intrinsic Matrix)
# Порог (threshold) задает максимальное отклонение в пикселях для RANSAC
E, mask = cv2.findEssentialMat(
    matched_pts_a,
    matched_pts_b,
    cameraMatrix=K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=3.0
)

# Фильтруем точки по маске RANSAC и раскладываем матрицу E на R и t
points, R, t, mask_pose = cv2.recoverPose(E, matched_pts_a, matched_pts_b, cameraMatrix=K, mask=mask)

# Фильтруем только инлайеры (наши 35 валидных точек)
valid_mask = mask.ravel() == 1
pts_a_inliers = matched_pts_a[valid_mask]
pts_b_inliers = matched_pts_b[valid_mask]

plot_matches(frames[0], frames[1], pts_a_inliers, pts_b_inliers)

cloud_sparse = triangulate_points_numpy(pts_a_inliers, pts_b_inliers, K, R, t)

height, width = frames[0].shape[:2]

# 1. Создаем пустые массивы для дисторсии (считаем, что линза идеальная)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 2. Исправленный вызов функции
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=K,
    distCoeffs1=dist_coeffs,   # Передаем пустой массив вместо None
    cameraMatrix2=K,
    distCoeffs2=dist_coeffs,   # Передаем пустой массив вместо None
    imageSize=(width, height),
    R=R,
    T=t.reshape(3, 1)          # Имя параметра должно быть строго 'T'
)

# Карты для первого кадра
map1_x, map1_y = cv2.initUndistortRectifyMap(K, None, R1, P1, (width, height), cv2.CV_32FC1)

# Карты для второго кадра
map2_x, map2_y = cv2.initUndistortRectifyMap(K, None, R2, P2, (width, height), cv2.CV_32FC1)

# Получаем ректифицированные (выровненные) кадры
rectified_frame_a = cv2.remap(frames[0], map1_x, map1_y, cv2.INTER_LINEAR)
# frame_b — это второй кадр, который мы брали для сопоставления (например, кадр 15)
rectified_frame_b = cv2.remap(frames[1], map2_x, map2_y, cv2.INTER_LINEAR)

plt.imshow(np.concatenate([rectified_frame_a, rectified_frame_b], axis=1))

# Считаем карту
disp = compute_disparity_numpy(rectified_frame_a, rectified_frame_b, window_size=7, max_disp=64)

# Чтобы увидеть её глазами, нормализуем для вывода на экран (0-255)
disp_visual = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("Disparity Map", disp_visual)
cv2.waitKey(0)

points_3d = cv2.reprojectImageTo3D(disp, Q)

# 2. Вытаскиваем цвета из ректифицированного левого кадра
# OpenCV читает в BGR, перевернем каналы в RGB для 3D-просмотрщиков
colors = cv2.cvtColor(rectified_frame_a, cv2.COLOR_BGR2RGB)

# 3. Создаем маску валидных точек
# Отрезаем пиксели, где диспаратность слишком маленькая (фон) или где улетело в бесконечность
mask = (disp > 0.5) & (disp < 64) & (np.isfinite(points_3d[:, :, 2]))

# 4. Схлопываем матрицы (H, W, 3) в плоские списки точек (N, 3) с помощью NumPy-маски
final_points = points_3d[mask]  # Массив размера (N, 3) с координатами X, Y, Z
final_colors = colors[mask]        # Массив размера (N, 3) с цветами R, G, B

save_ply_fast_numpy('myscene.ply', final_points, final_colors)

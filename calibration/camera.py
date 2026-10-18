import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==================== SETTINGS ====================
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.052
IMAGES_PATH = "infinite_focus/*jpg"
VERBOSE=False
# ==================================================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка идеальных 3D-точек
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []
valid_images = []

images = glob.glob(IMAGES_PATH)
if not images:
    print(f"Ошибка: Не найдено изображений по пути {IMAGES_PATH}")
    exit()

print(f"Найдено {len(images)} изображений для калибровки. Ищем углы...")

gray = None
img_size = None

# Включаем интерактивный режим отображения окон matplotlib
plt.ion()

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        valid_images.append(fname)
        print(f"[+] Успешно распознано: {os.path.basename(fname)}")

        # --- ВИЗУАЛИЗАЦИЯ 1: Первичная детекция углов ---
        if VERBOSE:
            img_vis = img.copy()
            cv2.drawChessboardCorners(img_vis, CHESSBOARD_SIZE, corners2, ret)
            img_vis_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            plt.figure("1. Детекция углов (закройте для продолжения)", figsize=(10, 7))
            plt.imshow(img_vis_rgb)
            plt.title(f"Детекция углов: {os.path.basename(fname)}")
            plt.axis('off')
            plt.show(block=True)
            plt.close()
    else:
        print(f"[-] Не удалось найти доску на кадре: {os.path.basename(fname)}")

if len(imgpoints) < 10:
    print("\nВнимание! Распознано слишком мало кадров (меньше 10). Результат может быть неточным.")

print("\nЗапуск математического движка калибровки...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print(f"Калибровка завершена! Ошибка репроекции (RMS): {ret:.4f} пикселей")

# Выключаем интерактивный режим для финальных окон
plt.ioff()

# --- ВИЗУАЛИЗАЦИЯ 2: Ошибка репроекции по кадрам ---
if VERBOSE:
    print("\nПроверка точности репроекции по кадрам...")
    frame_errors = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # Решаем проблему с несовпадением типов: приводим оба массива к плоскому виду (N, 2) типа float32
        pts_detected = imgpoints[i].reshape(-1, 2).astype(np.float32)
        pts_projected = imgpoints2.reshape(-1, 2).astype(np.float32)

        # Теперь cv2.norm отработает корректно
        error = cv2.norm(pts_detected, pts_projected, cv2.NORM_L2) / len(pts_projected)
        frame_errors.append(error)

        fname = valid_images[i]
        img = cv2.imread(fname)

        for pt_det, pt_proj in zip(pts_detected, pts_projected):
            x_det, y_det = map(int, pt_det)
            x_proj, y_proj = map(int, pt_proj)

            # Синий круг — детектор, Красный круг — математика, Зеленый отрезок — вектор ошибки
            cv2.circle(img, (x_det, y_det), 4, (255, 0, 0), -1)
            cv2.circle(img, (x_proj, y_proj), 2, (0, 0, 255), -1)
            cv2.line(img, (x_det, y_det), (x_proj, y_proj), (0, 255, 0), 1)

        cv2.putText(img, f"Frame Error: {error:.4f} px", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure("2. Точность репроекции (закройте для продолжения)", figsize=(10, 7))
        plt.imshow(img_rgb)
        plt.title(f"Репроекция: {os.path.basename(fname)} (Ошибка: {error:.4f} px)")
        plt.axis('off')
        plt.show(block=True)
        plt.close()

# --- ВИЗУАЛИЗАЦИЯ 3: Итоговый график ошибок ---
if VERBOSE:
    plt.figure("3. Итоговый график ошибок калибровки", figsize=(10, 5))
    plt.bar([os.path.basename(f) for f in valid_images], frame_errors, color='skyblue', edgecolor='black')
    plt.axhline(y=ret, color='r', linestyle='--', label=f'Mean RMS ({ret:.4f})')
    plt.axhline(y=1.0, color='g', linestyle=':', label='Threshold Max (1.0 px)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Ошибка репроекции (пиксели)')
    plt.title('Анализ ошибок калибровки по кадрам')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Исправленное сохранение JSON структуры
camera_data = {
    "resolution": {
        "width": img_size[0],
        "height": img_size[1]
    },
    "K": np.round(mtx, 2).tolist(),
    "D": np.round(dist, 6).flatten().tolist()
}

output_file = "camera.json"
with open(output_file, "w") as f:
    json.dump(camera_data, f, indent=4)

print(camera_data)
print(f"Параметры успешно записаны в файл: {output_file}")
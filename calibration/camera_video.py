# =====================================================================
#  calibrate_from_video.py
#  Калибровка на кадрах видео (не на jpg-снапшотах).
#  Проряжает видео, ищет шахматку, отсеивает блюр и дубликаты.
# =====================================================================
import cv2
import numpy as np
import json

# ==================== SETTINGS ====================
VIDEO_PATH      = 'realme11_focus_1m/VID_20261031_224915.mp4'   # ← ВАШЕ видео с шахматкой
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE     = 0.052
STEP            = 1          # каждый 4-й кадр
MIN_LAPLACIAN   = 50.0       # отсеять блюр (variance of Laplacian)
MIN_CORNER_SHIFT = 30.0      # px: дедупликация похожих кадров
OUTPUT_JSON     = "camera2.json"
# ==================================================

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Не удалось открыть {VIDEO_PATH}")

frame_idx = 0
objpoints, imgpoints = [], []
img_size = None
prev_corners = None
total_read = 0
total_checked = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total_read += 1

    if frame_idx % STEP != 0:
        frame_idx += 1
        continue
    frame_idx += 1
    total_checked += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]

    # фильтр по резкости (блюр → углы неточные)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < MIN_LAPLACIAN:
        continue

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if not found:
        continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # дедупликация: если углы почти там же, что в прошлом принятом кадре — пропуск
    if prev_corners is not None:
        shift = float(np.linalg.norm(
            corners.reshape(-1, 2) - prev_corners.reshape(-1, 2), axis=1
        ).mean())
        if shift < MIN_CORNER_SHIFT:
            continue

    objpoints.append(objp)
    imgpoints.append(corners)
    prev_corners = corners
    print(f"[+] кадр {frame_idx}: lap={lap_var:.0f}, принят")

cap.release()

print(f"\nПрочитано: {total_read}  Проверено (каждый {STEP}-й): {total_checked}")
print(f"Валидных кадров: {len(imgpoints)}")
if len(imgpoints) < 10:
    print("⚠️  Мало кадров — калибровка может быть неустойчивой")

# --- калибровка ---
result = cv2.calibrateCameraExtended(objpoints, imgpoints, img_size, None, None)
ret, mtx, dist, rvecs, tvecs, std_int, std_ext = result[:7]
std = np.asarray(std_int).ravel()

print(f"\nRMS: {ret:.4f} px")
print(f"fx = {mtx[0,0]:.2f} ± {std[0]:.2f}")
print(f"fy = {mtx[1,1]:.2f} ± {std[1]:.2f}")
print(f"cx = {mtx[0,2]:.2f} ± {std[2]:.2f}")
print(f"cy = {mtx[1,2]:.2f} ± {std[3]:.2f}")
print(f"k1 = {dist.ravel()[0]:.4f} ± {std[4]:.4f}")

# --- сохранить ---
camera_data = {
    "resolution": {"width": img_size[0], "height": img_size[1]},
    "K": np.round(mtx, 2).tolist(),
    "D": np.round(dist, 6).flatten().tolist(),
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(camera_data, f, indent=4)
print(f"\nСохранено в {OUTPUT_JSON}")
print(camera_data)
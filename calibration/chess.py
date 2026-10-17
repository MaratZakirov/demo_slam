import cv2
import numpy as np

SQUARE_SIZE_PX = 110
GRID_WIDTH = 10
GRID_HEIGHT = 7

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

image = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8) * 255

chess_width = GRID_WIDTH * SQUARE_SIZE_PX
chess_height = GRID_HEIGHT * SQUARE_SIZE_PX

MARGIN_X = (TARGET_WIDTH - chess_width) // 2
MARGIN_Y = (TARGET_HEIGHT - chess_height) // 2

for row in range(GRID_HEIGHT):
    for col in range(GRID_WIDTH):
        if (row + col) % 2 != 0:
            start_x = MARGIN_X + col * SQUARE_SIZE_PX
            start_y = MARGIN_Y + row * SQUARE_SIZE_PX
            end_x = start_x + SQUARE_SIZE_PX
            end_y = start_y + SQUARE_SIZE_PX
            image[start_y:end_y, start_x:end_x] = 0

cv2.imwrite('chess.jpg', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
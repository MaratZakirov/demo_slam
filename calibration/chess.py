import cv2
import numpy as np

SQUARE_SIZE_PX = 240
GRID_WIDTH = 10
GRID_HEIGHT = 7
MARGIN_PX = 100

img_width = GRID_WIDTH * SQUARE_SIZE_PX + (2 * MARGIN_PX)
img_height = GRID_HEIGHT * SQUARE_SIZE_PX + (2 * MARGIN_PX)

image = np.ones((img_height, img_width), dtype=np.uint8) * 255

for row in range(GRID_HEIGHT):
    for col in range(GRID_WIDTH):
        if (row + col) % 2 != 0:
            start_x = MARGIN_PX + col * SQUARE_SIZE_PX
            start_y = MARGIN_PX + row * SQUARE_SIZE_PX
            end_x = start_x + SQUARE_SIZE_PX
            end_y = start_y + SQUARE_SIZE_PX
            image[start_y:end_y, start_x:end_x] = 0

cv2.imwrite('chess.jpg', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"File chess.png")
print(f"Resolution: {img_width}x{img_height}")
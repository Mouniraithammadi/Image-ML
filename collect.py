import os
import cv2
import numpy as np

img = cv2.imread("image.jpg")

size = 224

height, width = img.shape[:2]

rows = int(np.ceil(height / size))
cols = int(np.ceil(width / size))

if rows * size > height:
    padding = np.zeros((rows * size - height, width, 3), dtype=np.uint8)
    img = np.vstack((img, padding))
if cols * size > width:
    padding = np.zeros((rows * size, cols * size - width, 3), dtype=np.uint8)
    img = np.hstack((img, padding))

squares = []
for r in range(rows):
    for c in range(cols):
        x = c * size
        y = r * size
        square = img[y:y+size, x:x+size, :]
        squares.append(square)

grid_size = (rows * size, cols * size, 3)
grid = np.zeros(grid_size, dtype=np.uint8)

for i, square in enumerate(squares):
    r = i // cols
    c = i % cols
    x = c * size
    y = r * size
    grid[y:y+size, x:x+size, :] = square

cv2.imwrite("results/rslt.jpg", grid)

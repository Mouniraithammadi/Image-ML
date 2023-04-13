import os
import cv2
import numpy as np

# Load the image
img = cv2.imread("your_image.jpg")

# Define the size of each square image
size = 224

# Get the height and width of the image
height, width = img.shape[:2]

# Calculate the number of rows and columns of the grid
rows = int(np.ceil(height / size))
cols = int(np.ceil(width / size))

# Pad the image if necessary to make it evenly divisible by size
if rows * size > height:
    padding = np.zeros((rows * size - height, width, 3), dtype=np.uint8)
    img = np.vstack((img, padding))
if cols * size > width:
    padding = np.zeros((rows * size, cols * size - width, 3), dtype=np.uint8)
    img = np.hstack((img, padding))

# Divide the image into squares and store them in a list
squares = []
for r in range(rows):
    for c in range(cols):
        x = c * size
        y = r * size
        square = img[y:y+size, x:x+size, :]
        squares.append(square)

# Create a new image to hold the grid of squares
grid_size = (rows * size, cols * size, 3)
grid = np.zeros(grid_size, dtype=np.uint8)

# Paste the squares into the grid
for i, square in enumerate(squares):
    r = i // cols
    c = i % cols
    x = c * size
    y = r * size
    grid[y:y+size, x:x+size, :] = square

# Save the grid image to a file
cv2.imwrite("results/grid.jpg", grid)

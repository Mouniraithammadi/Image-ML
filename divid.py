import cv2
import numpy as np
import os
# img = cv2.imread("img1.jpg")
def dividing(src):
    # Load the image
    img = cv2.imread(src)

    # Define the size of each square image
    size = 50

    # Get the height and width of the image
    height,width = img.shape[:2]

    # Calculate the number of rows and columns of the grid
    rows = int(np.ceil(height / size))
    cols = int(np.ceil(width / size))

    # Pad the image if necessary to make it evenly divisible by size
    if rows * size > height:
        padding = np.zeros((rows * size - height,width,3),dtype=np.uint8)
        img = np.vstack((img,padding))
    if cols * size > width:
        padding = np.zeros((rows * size,cols * size - width,3),dtype=np.uint8)
        img = np.hstack((img,padding))
    if not os.path.exists("results"):
        os.makedirs("results")
    # Divide the image into squares
    for r in range(rows):
        for c in range(cols):
            x = c * size
            y = r * size
            square = img[y:y + size,x:x + size,:]
            filename = f"results/square_{r}_{c}.jpg"
            cv2.imwrite(filename,square)

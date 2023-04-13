import cv2
import numpy as np
import os
from build import get_similar_emoji
image = cv2.imread('img1.jpg') # replace 'image.jpg' with the path to your image file
square_size = 32 # replace 32 with the desired size of each square in pixels


for y in range(0, image.shape[0], square_size):

    for x in range(0, image.shape[1], square_size):
        square = image[y:y+square_size, x:x+square_size]
        # Perform image processing or feature extraction on 'square' if needed
        # Call your 'get_similarity' function to get the path of the most similar object (emoji)
        most_similar_object_path = get_similar_emoji(square)
        # Load the most similar object (emoji) using OpenCV
        emoji = cv2.imread(most_similar_object_path)
        emoji = cv2.resize(emoji,(square_size,square_size))
        # Replace the square in the original image with the emoji
        image[y:y+square_size, x:x+square_size] = emoji
cv2.imwrite('output.jpg', image) # replace 'output.jpg' with the desired output image filename


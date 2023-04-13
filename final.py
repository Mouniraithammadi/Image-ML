import cv2
import numpy as np
import os
from build import get_similar_emoji
image = cv2.imread('img1.jpg') 
square_size = 32 # or 16 , 8 , 4 , 64 ,...


for y in range(0, image.shape[0], square_size):

    for x in range(0, image.shape[1], square_size):
        square = image[y:y+square_size, x:x+square_size]
       
        most_similar_object_path = get_similar_emoji(square)
        emoji = cv2.imread(most_similar_object_path)
        emoji = cv2.resize(emoji,(square_size,square_size))
        image[y:y+square_size, x:x+square_size] = emoji
cv2.imwrite('output.jpg', image) 


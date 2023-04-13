import pinecone
import numpy as np
import yaml

with open('//content//drive//MyDrive//img Simulator//config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
pinecone.init(api_key=config["api_key"],environment=config["environment"])
index = pinecone.Index(index_name=config["index_name"])

def put(ids, vectors):
    ids_arr = np.array(ids)
    ids_arr = ids_arr[:,np.newaxis]
    vectors_flat = [np.array(values).flatten() for values in vectors]  # Flatten vectors
    arr = np.concatenate([ids_arr,vectors_flat],axis=1)
    vectors_list = [(str(id_),tuple(values)) for id_,*values in arr]
    return index.upsert(vectors=vectors_list)

def get(Vector):
    ndarray = np.array(Vector)
    # Convert to list
    vector = ndarray.tolist()
    res = index.query(queries=[vector],top_k=1)
    return res

import cv2
import os
# from pincone import put
from keras.models import Model
from keras import applications
vgg16 = applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

def get_feature_vector(img):
 img1 = cv2.resize(img, (224, 224))
 feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
 return feature_vector
def read_image(file_path):
 img = cv2.imread(file_path)
 return img


# from pincone import get
# define the size of each square
square_size = 32
import os
image_dir = "//content//drive//MyDrive//img Simulator//images"

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

def get_similar_emoji(square):
    vector = get_feature_vector(square)
    info = get(vector)
    index = str(info["results"][0]["matches"][0]["id"])
    # print(image_files[int(index)])
    print(index)
    return index

import cv2
import numpy as np
import os

image = cv2.imread('emoji_red_heart_full.jpg') # replace 'image.jpg' with the path to your image file
square_size = 64 # replace 32 with the desired size of each square in pixels

com = 0 
for y in range(0, image.shape[0], square_size):

    for x in range(0, image.shape[1], square_size):
        # square = image[y:y+square_size, x:x+square_size]
        # # Perform image processing or feature extraction on 'square' if needed
        # # Call your 'get_similarity' function to get the path of the most similar object (emoji)
        # most_similar_object_path = get_similar_emoji(square)
        # # Load the most similar object (emoji) using OpenCV
        # emoji = cv2.imread(most_similar_object_path, cv2.IMREAD_COLOR)
        # emoji = cv2.resize(emoji,(square_size,square_size))
        # # Replace the square in the original image with the emoji
        # image[y:y+square_size, x:x+square_size] = emoji
        square = image[y:y+square_size, x:x+square_size]
        # Perform image processing or feature extraction on 'square' if needed
        # Call your 'get_similarity' function to get the path of the most similar object (emoji)
        most_similar_object_path = get_similar_emoji(square)
        # Load the most similar object (emoji) using OpenCV
        emoji = cv2.imread(most_similar_object_path, cv2.IMREAD_COLOR) # Load the emoji image without alpha channel
        # print("Emoji shape:", emoji.shape)
        # print("Square shape:", square.shape)
        # Resize the emoji image to match the size of the square region
        emoji = cv2.resize(emoji, (square.shape[1], square.shape[0]))
        # Replace the square in the original image with the emoji
        image[y:y+square_size, x:x+square_size] = emoji
        print(com)
        com += 1
cv2.imwrite('output2.jpg', image) # replace 'output.jpg' with the desired output image filename

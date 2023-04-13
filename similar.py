import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
from astropy.stats import spatial

from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
import tensorflow as tf
from keras import applications
from scipy.spatial.distance import cosine
vgg16 = applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)

def get_feature_vector(img):

 img1 = cv2.resize(img, (224, 224))
 feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
 return feature_vector


# def calculate_similarity(vector1, vector2):
#  return 1- spatial.distance.cosine(vector1, vector2)


def read_image(file_path):
 img = cv2.imread(file_path)
 return img


def calculate_similarity(vector1,vector2):

 flattened_vector1 = np.ravel(vector1)
 flattened_vector2 = np.ravel(vector2)
 return 1 - cosine(flattened_vector1,flattened_vector2)


img1 = read_image("/content/img Simulator/img1.jpg")
img2 = read_image("/content/img Simulator/imgs/img2.jpg")
img3 = read_image("/content/img Simulator/imgs/img4.jpeg")
f1 = get_feature_vector(img1)
f2 = get_feature_vector(img2)
f3 = get_feature_vector(img3)
k = 0 
for i in f1[0]:
  print(str(k)+" > "+str(i))
  k+=1
# print(f1)
# print(f2)
# print(f3)
# print("check me1 with me2 ",calculate_similarity(f1, f2))
# print("check me1 with tome ",calculate_similarity(f1, f3))
# print("check me2 with tome ",calculate_similarity(f2, f3))



# # compare with many imgs
# import os
# import pickle

# # Set up a list of image file paths
# image_dir = "imgs"
# image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".jpeg")]

# # # Precompute the feature vectors for all images and store them in a dictionary
# # feature_vectors = {}
# # for image_file in image_files:
# #     img = read_image(image_file)
# #     feature_vector = get_feature_vector(img)
# #     feature_vectors[image_file] = feature_vector
# #
# # # Store the feature vectors dictionary in a file for later use
# # with open("feature_vectors.pkl", "wb") as f:
# #     pickle.dump(feature_vectors, f)
# #
# # # Load the feature vectors from the file
# # with open("feature_vectors.pkl", "rb") as f:
# #     feature_vectors = pickle.load(f)
# #
# # # Compare a new image with the precomputed images and print the top 5 matches
# # new_image_file = "img1.jpg"
# # new_image = read_image(new_image_file)
# # new_feature_vector = get_feature_vector(new_image)
# #
# # similarities = {}
# # for image_file, feature_vector in feature_vectors.items():
# #     similarity = calculate_similarity(new_feature_vector, feature_vector)
# #     similarities[image_file] = similarity
# #
# # top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
# # for match in top_matches:
# #     print(match[0], match[1])
# #     print(match[0], match[1])

# # Read the images into a numpy array and preprocess them for VGG16
# batch_size = 32
# num_batches = int(np.ceil(len(image_files) / batch_size))
# images = []
# for i in range(num_batches):
#     batch_files = image_files[i*batch_size:(i+1)*batch_size]
#     batch_images = [read_image(f) for f in batch_files]
#     batch_images = [cv2.resize(img, (224, 224)) for img in batch_images]
#     batch_images = np.array(batch_images, dtype="float32")
#     batch_images = applications.vgg16.preprocess_input(batch_images)
#     images.append(batch_images)

# # Pass the images through the VGG16 model in batches
# features = []
# for i in range(num_batches):
#     batch_features = basemodel.predict(images[i])
#     features.append(batch_features)

# # Concatenate the features into a single array
# features = np.concatenate(features, axis=0)
# # Compute the similarity scores between the reference image and all other images
# ref_index = 0 # Index of the reference image in the list of image files
# ref_features = features[ref_index]
# similarity_scores = []
# for i in range(len(features)):
#     if i == ref_index:
#         similarity_scores.append(1.0) # Set the similarity score to 1.0 for the reference image itself
#     else:
#         score = calculate_similarity(ref_features, features[i])
#         similarity_scores.append(score)

# # Print the similarity scores
# for i in range(len(similarity_scores)):
#     print("Similarity score between image %d and the reference image: %.4f" % (i, similarity_scores[i]))

import cv2
import os
import time 
from keras.models import Model
from keras import applications
vgg16 = applications.VGG16(weights='imagenet', include_top=True, pooling='max', input_shape=(224, 224, 3))
basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer('fc2').output)
import pinecone
import numpy as np
import yaml

with open('//content//drive//MyDrive//img Simulator//config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
pinecone.init(api_key=config["api_key"],environment=config["environment"])
index = pinecone.Index(index_name=config["index_name"])

def get_feature_vector(img):
 img1 = cv2.resize(img, (224, 224))
 feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
 return feature_vector


def read_image(file_path):
 img = cv2.imread(file_path)
 return img

image_dir = "//content//drive//MyDrive//img Simulator//images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
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

def start():
    ids =[]
    vectors = []
    k = 0 
    id = [str(img) for img in image_files]  # Simplify ID creation
    vector = [list(get_feature_vector(read_image(img))) for img in image_files]
    print(len(id))
    for j in range(21):
      for i in range(45):
        k = i+(j*45)
        if k == len(id):
          break
        ids.append(id[k])
        vectors.append(vector[k])
      print(put(ids, vectors))
      if k == len(id):
          break
      ids =[]
      vectors = []
    print(len(ids))
      
    
      
    
    
# start()

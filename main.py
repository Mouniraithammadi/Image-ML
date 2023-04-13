import cv2
import os
from pincone import put
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

image_dir = "images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]


def start():
    ids = []
    vectors = []
    id = [str(img) for img in image_files]  # Simplify ID creation
    # vector = [list(get_feature_vector(read_image(img))) for img in image_files]
    for i in range(64,len(id)):
        ids.append(id[i])
        # vectors.append(vector[i])

    print(len(ids))
    print(len(id))
    # print(put(ids,vectors))

# start()
print((4))
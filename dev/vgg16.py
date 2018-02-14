from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

directory_in_str = "../../work/NYTImesWorld/"

directory = os.fsencode(directory_in_str)

model = VGG16(weights='imagenet', include_top=False)

features = []

for img in os.listdir(directory):
    img_name = os.fsdecode(img)
    img_path = os.path.join(directory, img_name)
    pic = image.load_img(img_path, target_size=(2048, 1365))
    x = image.img_to_array(pic)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features.append(model.predict(x))

print(features.head(10))

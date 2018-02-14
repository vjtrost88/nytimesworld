from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

target_dir = "../../work/NYTimesWorld/"

model = VGG16(weights='imagenet', include_top=False)

features = []

for img_path in images:
    img = image.load_img(img_path, target_size=(2048, 1365))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features.append(model.predict(x))

print(features.head(10))

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd

model = VGG16(weights='imagenet', include_top=False)

f = open("file-paths.txt", "r")
contents = f.readlines()
contents = [line.rstrip('\n') for line in contents]

features = []

for img_path in contents:

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features.append(model.predict(x))

#features = pd.DataFrame(features)
print(features[1:5])

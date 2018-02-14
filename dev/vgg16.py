import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def extract_features_with_vgg16(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).flatten()
    return features

def main():
    f = open("file-paths-r10.txt", "r")
    contents = f.readlines()
    contents = [line.rstrip('\n') for line in contents]
    model = VGG16(weights='imagenet', include_top=False)
    feature_map = []

    for img_path in contents:
        features = extract_features_with_vgg16(model, img_path)
        feature_map.append(features)

    feature_map = np.matrix(feature_map)
    np.savetxt("test.csv", feature_map, delimiter=',')

if __name__ == '__main__':
    main()

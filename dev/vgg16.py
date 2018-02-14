from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from keras.preprocessing import image
from places_utils import preprocess_input

target_dir = "../../work/NYTimesWorld/"
batch_size = 32

model = VGG16_Hybrid_1365(weights='places', include_top=False)

features = []

for img_path in images:
    img = image.load_img(img_path, target_size=(2048, 1365))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features.append(model.predict(x))

print(features.head(10))

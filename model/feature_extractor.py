import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# model = InceptionV3(weights='imagenet')
# print("Model loaded successfully")

def load_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    return Model(base_model.input, base_model.layers[-2].output)

def preprocess_image(img_pil):
    img = img_pil.resize((299, 299))
    img = np.array(img)
    if img.shape[-1] == 4:  # remove alpha if present
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def extract_features(img_pil, model):
    img = preprocess_image(img_pil)
    feature = model.predict(img, verbose=0)
    return feature.reshape((1, 2048))

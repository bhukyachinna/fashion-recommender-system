import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50V2(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = load_img(img_path, target_size = (224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis = 0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        indices = recommend(features,feature_list)
        # show
        cols = st.columns(5)
        enum_cols = enumerate(cols)

        for i, col in enum_cols:
            with col:
                img = Image.open(filenames[indices[0][i]])
                img = img.resize((450, 600))
                st.image(img)
    else:
        st.header("Some error occured in file upload.")

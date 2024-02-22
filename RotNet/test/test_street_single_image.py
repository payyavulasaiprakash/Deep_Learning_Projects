from __future__ import print_function

import os,tqdm
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import decode_predictions
sys.path.append('..')
from utils import display_examples, RotNetDataGenerator, angle_error
from data.street_view_ours import get_filenames

def save_text(_file, content):
    with open(_file, mode='w+', encoding='utf-8') as f:
        f.write(content)   



def load_and_preprocess_image(image_path):
    # Load the image using PIL (Python Imaging Library)
    image = Image.open(image_path)
    
    # Resize the image to match the input shape expected by the model
    # Adjust the dimensions as needed based on your model's input shape
    target_size = (224, 224)
    image = image.resize(target_size)
    
    # Convert the image to a numpy array and normalize its values
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    # You may need to apply additional preprocessing steps depending on your model
    # For example, mean subtraction, channel reordering, etc.
    # Ensure that the preprocessing steps match those used during training
    
    return image_array


def generate_csv(angle):
    csv = 'angle \n'
    for x in angle:
        csv = csv + str(x) + '\n'
    save_text('{}/{}.csv'.format(output_folder,'top'), csv)


street_view_dir = ''
output_folder = '' 
test_filenames = get_filenames(street_view_dir,0.9999999)[0]
print('test_filenames',len(test_filenames))
print(test_filenames)
detected_scores = []

def predict_single_image(image_path, model):
    # Load the image and preprocess it
    image = load_and_preprocess_image(image_path)
    # Reshape the image to match the expected input shape of the model
    image = np.expand_dims(image, axis=0)
    # Perform model prediction
    prediction = model.predict(image)
    print(prediction)
    # Return the prediction
    return prediction

test_data = RotNetDataGenerator(
            test_filenames,
            input_shape=(224, 224, 3),
            batch_size=1,
            preprocess_func=preprocess_input,
            crop_center=False,
            crop_largest_rect=False,
            shuffle=False,
            rotate=False
        )

angles = []
models_folder = ''
for model_file in os.listdir(models_folder):
    model_location = os.path.join(models_folder, model_file)
    model = load_model(model_location, custom_objects={'angle_error': angle_error})
    k=0
    for test_image in tqdm.tqdm(test_data,desc='Progress'):
        pred=list(model.predict(test_image[0]).squeeze())
        angles.append(pred.index(max(pred)))
        k+=1
        if k==2299:
            generate_csv(angles)


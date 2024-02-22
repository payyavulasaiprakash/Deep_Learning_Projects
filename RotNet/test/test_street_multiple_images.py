from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

sys.path.append('..')
from utils import display_examples, RotNetDataGenerator, angle_error
from data.street_view_ours import get_filenames

def save_text(_file, content):
    with open(_file, mode='w+', encoding='utf-8') as f:
        f.write(content)   

def generate_csv(model_file, to_be_added, output_folder):
    csv = 'model_file, test_loss, test_angle_error \n'
    for x in to_be_added:
        _csv = ','.join(x)
        csv = csv + _csv + '\n'
    save_text('{}/{}.csv'.format(output_folder,model_file), csv)


street_view_dir = '' #folder in which images are there
output_folder = '' #to save output csv in folder
test_filenames = get_filenames(street_view_dir,0.9999999)[0] #0.9999999 percentage of test data to take for testing
print('test_filenames',len(test_filenames))
print(test_filenames)
to_be_added = []
models_folder = '' #models_folder_path
model_files = os.listdir(models_folder)
for model_file in model_files:
    model_location = os.path.join(models_folder, model_file)
    model = load_model(model_location, custom_objects={'angle_error': angle_error})
    
    batch_size = 64
    out = model.evaluate_generator(
        RotNetDataGenerator(
            test_filenames,
            input_shape=(224, 224, 3),
            batch_size=batch_size,
            preprocess_func=preprocess_input,
            crop_center=False,
            crop_largest_rect=False,
            shuffle=False,
            rotate=False
        ),
        steps=len(test_filenames) / batch_size
    )

    print('Test loss:', out[0])
    print('Test angle error:', out[1])
    to_be_added.append([model_file, str(out[0]), str(out[1])])


generate_csv(model_file, to_be_added, output_folder)

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

def generate_csv(model_file, detected, output_folder):
    # folder_name = folder.split('/')[-1]
    csv = 'model_file, test_loss, test_angle_error \n'
    for x in detected:
        _csv = ','.join(x)
        csv = csv + _csv + '\n'

    # for x in not_detected:
    #     _csv = ','.join(x)
    #     csv = csv + _csv + '\n'

    save_text('{}/{}.csv'.format(output_folder,1), csv)


street_view_dir = '/home/vishwam/mountpoint/saiprakash/face_orientation_new/_0_sample'
output_folder = '/home/vishwam/mountpoint/saiprakash/face_orientation_new/train/jbp_jan_55_75_vgg_sample_frb_22_testing'
test_filenames = get_filenames(street_view_dir,0.9999999)[0]
print('test_filenames',len(test_filenames))
print(test_filenames)
detected_scores = []
# model_location = os.path.join('/home/vishwam/mountpoint/saiprakash/face_orientation_new/train/jbp_jan_55_75_sample_models', 'rotnet_street_view_resnet50_01-val_acc_85.5484.h5')
# model = load_model(model_location, custom_objects={'angle_error': angle_error})
model_files = os.listdir('/home/vishwam/mountpoint/saiprakash/face_orientation_new/train/jbp_jan_55_75_vgg_sample_frb_22')
for model_file in model_files:
    model_location = os.path.join('/home/vishwam/mountpoint/saiprakash/face_orientation_new/train/jbp_jan_55_75_vgg_sample_frb_22', model_file)
    model = load_model(model_location, custom_objects={'angle_error': angle_error})
    
    batch_size = 64
    out = model.evaluate_generator(
        RotNetDataGenerator(
            test_filenames,
            input_shape=(224, 224, 3),
            batch_size=batch_size,
            preprocess_func=preprocess_input,
            crop_center=True,
            crop_largest_rect=True,
            shuffle=True
        ),
        steps=len(test_filenames) / batch_size
    )

    print('Test loss:', out[0])
    print('Test angle error:', out[1])
    detected_scores.append([model_file, str(out[0]), str(out[1])])
    # generate_csv(model_file, detected_scores, output_folder)

    num_images = 5

    display_examples(
        model, 
        test_filenames,
        num_images=num_images,
        size=(224, 224),
        crop_center=True,
        crop_largest_rect=True,
        preprocess_func=preprocess_input,
    )
generate_csv(model_file, detected_scores, output_folder)
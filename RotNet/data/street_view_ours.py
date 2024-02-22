from __future__ import print_function

import os
import wget
import zipfile

def get_filenames(path,train_size=0.9):

    image_paths = []
    for filename in os.listdir(path):
        image_paths.append(os.path.join(path, filename))

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * train_size)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = image_paths[n_train_samples:]

    return train_filenames, test_filenames

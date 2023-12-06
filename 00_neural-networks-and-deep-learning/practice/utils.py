import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from fastai.vision.all import *

def load_dataset(folder_path='bird_or_not', image_size=(64, 64)):
    categories = ['bird', 'forest']
    images = []
    labels = []

    # Iterate over categories
    for label, category in enumerate(categories):
        category_path = os.path.join(folder_path, category)
        for file in os.listdir(category_path):
            # Load and resize the image
            img_path = os.path.join(category_path, file)
            image = imread(img_path)
            image = resize(image, image_size, anti_aliasing=True)
            
            # Append to dataset
            images.append(image)
            labels.append(label)

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split into training and testing sets
    split_idx = int(0.8 * len(images))  # 80% for training, 20% for testing
    train_set_x_orig = images[:split_idx]
    train_set_y_orig = labels[:split_idx]
    test_set_x_orig = images[split_idx:]
    test_set_y_orig = labels[split_idx:]

    # Reshape labels
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Classes array
    classes = np.array(categories)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


import os
import numpy as np
from PIL import Image

def load_and_resize_images():
    directory = 'cat_or_not'
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize((64, 64))
            img_array = np.array(img)
            images.append(img_array)
    return images

bird_images = load_and_resize_images('cat_or_not/bird')
forest_images = load_and_resize_images('cat_or_not/forest')

all_images = np.array(bird_images + forest_images)
print(all_images.shape)

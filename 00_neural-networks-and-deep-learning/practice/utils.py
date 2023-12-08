import numpy as np
from PIL import Image
import os

def load_dataset2(directory, seed=None, image_size=(64, 64)):
    train_set_x_orig = []
    train_set_y = []
    test_set_x_orig = []
    test_set_y = []
    classes = ['forest', 'bird']

    # Set the seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    for index, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        all_images = os.listdir(class_dir)

        # Split data into training and testing (e.g., 80% training, 20% testing)
        split_index = int(0.8 * len(all_images))
        train_images = all_images[:split_index]
        test_images = all_images[split_index:]

        for image_name in train_images:
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)  # Resize to 64x64 or any desired size
            train_set_x_orig.append(np.array(image))
            train_set_y.append(index)

        for image_name in test_images:
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path)
            image = image.resize(image_size)
            test_set_x_orig.append(np.array(image))
            test_set_y.append(index)

    # Combine images and labels into tuples for training and testing sets
    combined_train = list(zip(train_set_x_orig, train_set_y))
    combined_test = list(zip(test_set_x_orig, test_set_y))

    # Shuffle the combined lists
    np.random.shuffle(combined_train)
    np.random.shuffle(combined_test)

    # Separate the images and labels after shuffling
    train_set_x_orig, train_set_y = zip(*combined_train)
    test_set_x_orig, test_set_y = zip(*combined_test)

    # Convert them back to numpy arrays
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y = np.array(train_set_y).reshape((1, -1))
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y = np.array(test_set_y).reshape((1, -1))

    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, np.array(classes)


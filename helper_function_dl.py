import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt




def split_image_data(source_dir, train_dir, test_dir, train_ratio=0.8):
    """
    Splits image data into training and testing datasets based on the given ratio, retaining the original folder.

    Parameters:
    - source_dir (str): Path to the source directory containing subdirectories of images (one subdirectory per class).
    - train_dir (str): Path to the directory where training data will be stored.
    - test_dir (str): Path to the directory where testing data will be stored.
    - train_ratio (float): Ratio of data to be used for training (e.g., 0.8 for 80% training and 20% testing).

    Behavior:
    - Each subdirectory in the source directory is treated as a class.
    - Files from each class are randomly shuffled and split into training and testing sets.
    - Subfolders for each class are created in the train_dir and test_dir.
    - Original files remain intact in the source directory.

    Example:
        split_image_data('dataset', 'dataset/train', 'dataset/test', 0.8)
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            total_files = len(files)

            if total_files == 0:
                print(f"Skipping empty directory: {subfolder_path}")
                continue

            random.shuffle(files)
            split_index = int(total_files * train_ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]

            train_subfolder = os.path.join(train_dir, subfolder)
            test_subfolder = os.path.join(test_dir, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)

            for file_name in train_files:
                shutil.copy(os.path.join(subfolder_path, file_name), os.path.join(train_subfolder, file_name))
            for file_name in test_files:
                shutil.copy(os.path.join(subfolder_path, file_name), os.path.join(test_subfolder, file_name))

    print("Data split completed.")


import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def show_random_images(data_dir):
    """
    Display random images from up to 5 random classes within a dataset directory.

    Parameters:
    - data_dir (str): Path to the dataset directory. Each subdirectory is treated as a class, 
                      and the function will randomly select one image from up to 5 classes.

    Behavior:
    - Ignores `.ipynb_checkpoints` and other non-directory files.
    - If the dataset contains more than 5 classes, a random subset of 5 classes is selected.
    - One random image is displayed from each selected class.
    - The images are displayed in a horizontal layout with their class name and file name as titles.

    Example:
        show_random_images('path/to/your/dataset')
    """
    # List all valid class directories (exclude '.ipynb_checkpoints')
    class_names = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d != '.ipynb_checkpoints'
    ]

    if not class_names:
        print("No valid class directories found in the dataset.")
        return

    # Select up to 5 random classes
    if len(class_names) > 5:
        class_names = random.sample(class_names, 5)

    images = []
    titles = []

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        # Get all files in the class directory
        files = os.listdir(class_path)
        if not files:
            print(f"Class directory '{class_name}' is empty. Skipping.")
            continue
        # Randomly select an image file
        image_file = random.choice(files)
        image_path = os.path.join(class_path, image_file)
        try:
            image = Image.open(image_path)
            images.append(image)
            titles.append(f"Class: {class_name}\nFile: {image_file}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

    if not images:
        print("No images found to display.")
        return

    # Display images in a horizontal layout
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    if len(images) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, pad=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def checks_and_remove_invalid_images(base_directory, valid_extensions={".jpg", ".jpeg", ".png", ".gif", ".bmp"}, remove=False):
    """
    Iterates through a directory with class folders, checks if images are in valid formats,
    records invalid images, and optionally removes them.

    Parameters:
    - base_directory (str): Path to the directory containing class folders.
    - valid_extensions (set): Set of valid file extensions (default is common image formats).
    - remove (bool): Whether to remove invalid images (default is False).

    Returns:
    - list: A list of tuples with invalid image names and their respective class folders.

    Example:
        check_and_remove_invalid_images('dataset', remove=True)
    """
    invalid_images = []

    # Normalize valid_extensions to lowercase
    valid_extensions = {ext.lower() for ext in valid_extensions}

    for class_folder in os.listdir(base_directory):
        class_folder_path = os.path.join(base_directory, class_folder)
        
        if os.path.isdir(class_folder_path):
            for file_name in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, file_name)
                file_extension = os.path.splitext(file_name)[-1].lower()

                if file_extension in valid_extensions:
                    try:
                        # Open the image with Pillow
                        with Image.open(file_path) as img:
                            # Check if the image format is valid
                            image_format = img.format.lower()
                            
                    
                            if image_format not in [ext.strip(".") for ext in valid_extensions]:
                                raise ValueError(f"Invalid format: {image_format}")

                            # Verify the image's structure
                            img.verify()  # This checks the file's integrity

                    except Exception as e:
                        # Flag the image as invalid if any step fails
                        invalid_images.append((file_name, class_folder))
                        if remove:
                            os.remove(file_path)
                            print(f"Removed invalid image: {file_name} in folder {class_folder} - {e}")
                else:
                    # Flag the image as invalid due to format mismatch
                    invalid_images.append((file_name, class_folder))
                    if remove:
                        os.remove(file_path)
                        print(f"Removed invalid format image: {file_name} in folder {class_folder}")

    print("Invalid images found:", invalid_images)
    return invalid_images


def build_model(model, excluded_layers=0):
    """
    Modify a pre-trained model to set the specified layers as trainable.

    Parameters:
    - model: Pre-trained model to modify (e.g., a model from TensorFlow/Keras).
    - excluded_layers: Number of initial layers to set as trainable.
      If 0, all layers are frozen (non-trainable).

    Returns:
    - Modified model with layers frozen/trainable as specified.
    """
    
    # Freeze all layers initially
    model.trainable = False  

    if excluded_layers > 0:
        # Make the first `excluded_layers` trainable
        for i in range(excluded_layers):  # Iterate through the desired layers
          model.layers[i].trainable = True  # Set them to trainable
          
    return model

    import matplotlib.pyplot as plt

def plot_model_comparison(models, history_list, metric='accuracy'):
    """
    Plots the specified metric (accuracy or loss) for multiple models.

    Args:
        models: A list of model names (strings).
        history_list: A list of model history objects (returned by model.fit).
        metric: The metric to plot ('accuracy' or 'loss').
    """

    plt.figure(figsize=(12, 6))  # Adjust figure size as needed

    for i, model_name in enumerate(models):
        history = history_list[i]
        
        if metric == 'accuracy':
            plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
            plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
        elif metric == 'loss':
            plt.plot(history.history['loss'], label=f'{model_name} Train Loss')
            plt.plot(history.history['val_loss'], label=f'{model_name} Validation Loss')

    plt.title(f'Model Comparison ({metric.capitalize()})')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, test_data):
    """Plots a confusion matrix for the given model and test data.

    Args:
        model: The trained Keras model.
        test_data: The test dataset (a `tf.data.Dataset` object).
    """

    # Get predictions and true labels
    y_pred_probs = model.predict(test_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_true = np.argmax(y_true, axis=1)

    # Infer class names from the test data
    num_classes = y_pred_probs.shape[1]
    class_names = [f"Class {i}" for i in range(num_classes)]

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    # Add labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Show the plot
    plt.show()

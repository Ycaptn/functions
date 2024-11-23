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


def show_random_images(data_dir):
    """
    Display random images from up to 5 random classes within a dataset directory.

    Parameters:
    - data_dir (str): Path to the dataset directory. Each subdirectory is treated as a class, 
                      and the function will randomly select one image from up to 5 classes.

    Behavior:
    - If the dataset contains more than 5 classes, a random subset of 5 classes is selected.
    - One random image is displayed from each selected class.
    - The images are displayed in a horizontal layout with their class name and file name as titles.

    Example:
        show_random_images('path/to/your/dataset')
    """
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if len(class_names) > 5:
        class_names = random.sample(class_names, 5)

    images = []
    titles = []

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        image_file = random.choice(os.listdir(class_path))
        image_path = os.path.join(class_path, image_file)
        image = Image.open(image_path)
        images.append(image)
        titles.append(f"Class: {class_name}\nFile: {image_file}")

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
    Modify a pre-trained model to set the first `excluded_layers` as non-trainable.

    Parameters:
    - model: Pre-trained model to modify (e.g., a model from TensorFlow/Keras).
    - excluded_layers: Number of initial layers to freeze (non-trainable).
      If 0, all layers are frozen.

    Returns:
    - Modified model with layers frozen as specified.
    """
    if excluded_layers == 0:
        # Freeze all layers
        for layer in model.layers:
            layer.trainable = False
    else:
        # Freeze the first `excluded_layers`
        for layer in model.layers[:excluded_layers]:
            layer.trainable = False

    # Recompile the model after freezing layers
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

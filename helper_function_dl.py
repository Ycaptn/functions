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
    # Ensure train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through each subfolder (class)
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # List all files in the subfolder
            files = os.listdir(subfolder_path)
            total_files = len(files)
            
            if total_files == 0:
                print(f"Skipping empty directory: {subfolder_path}")
                continue
            
            # Shuffle and split files into training and testing sets
            random.shuffle(files)
            split_index = int(total_files * train_ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]
            
            # Create corresponding train and test subfolders
            train_subfolder = os.path.join(train_dir, subfolder)
            test_subfolder = os.path.join(test_dir, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)
            
            # Copy files to train and test folders
            for file_name in train_files:
                shutil.copy(os.path.join(subfolder_path, file_name), os.path.join(train_subfolder, file_name))
            for file_name in test_files:
                shutil.copy(os.path.join(subfolder_path, file_name), os.path.join(test_subfolder, file_name))
                
    print("Data split completed.")



def show_random_images(data_dir):
    """

    Display random images from up to 5 random classes within a dataset directory.

    Parameters:
    data_dir (str): Path to the dataset directory. Each subdirectory is treated as a class, 
                    and the function will randomly select one image from up to 5 classes.

    Behavior:
    - If the dataset contains more than 5 classes, a random subset of 5 classes is selected.
    - One random image is displayed from each selected class.
    - The images are displayed in a horizontal layout with their class name and file name as titles.

    Example usage: 
    show_random_images('path/to/your/dataset')

    """
    # Get list of class folders
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Limit to a maximum of 5 classes
    if len(class_names) > 5:
        class_names = random.sample(class_names, 5)  # Randomly select 5 classes

    images = []  # To store images
    titles = []  # To store titles for display

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        
        # Choose a random image file within the class folder
        image_file = random.choice(os.listdir(class_path))
        image_path = os.path.join(class_path, image_file)
        
        # Load the image
        image = Image.open(image_path)
        images.append(image)
        
        # Create a title with class and file information
        titles.append(f"Class: {class_name}\nFile: {image_file}")

    # Display all selected images in a horizontal layout
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))  # Dynamic sizing for the number of images

    if len(images) == 1:  # Handle case when there is only one directory
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, pad=10)
        ax.axis('off')  # Hide axes for better visualization

    plt.tight_layout()
    plt.show()



def check_and_remove_invalid_images(base_directory, valid_extensions={".jpg", ".jpeg", ".png", ".gif", ".bmp"}, remove=False):
    """
    Iterates through a directory with class folders, checks if images are in valid formats,
    records invalid images, and optionally removes them.

    Args:
        base_directory (str): Path to the directory containing class folders.
        valid_extensions (set): Set of valid file extensions (default is common image formats).
        remove (bool): Whether to remove invalid images (default is False).

    Returns:
        list: A list of tuples with invalid image names and their respective class folders.
    """
    invalid_images = []

    for class_folder in os.listdir(base_directory):
        class_folder_path = os.path.join(base_directory, class_folder)
        
        if os.path.isdir(class_folder_path):  # Ensure it's a directory
            for file_name in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, file_name)
                file_extension = os.path.splitext(file_name)[-1].lower()

                if file_extension in valid_extensions:  # Check if extension is valid
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verify image content
                    except Exception as e:
                        # Record invalid image
                        invalid_images.append((file_name, class_folder))
                        if remove:
                            os.remove(file_path)
                            print(f"Removed corrupted image: {file_name} in folder {class_folder} - {e}")
                else:
                    # Record invalid format image
                    invalid_images.append((file_name, class_folder))
                    if remove:
                        os.remove(file_path)
                        print(f"Removed invalid format image: {file_name} in folder {class_folder}")

    print("Invalid images found:", invalid_images)
    return invalid_images


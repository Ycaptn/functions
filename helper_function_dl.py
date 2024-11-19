import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt


def split_image_data(source_dir, train_dir, test_dir, train_ratio):
    # Ensure train and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get each subfolder in the source directory
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # List all files in the subfolder
            files = os.listdir(subfolder_path)
            total_files = len(files)
            
            # Shuffle and split files into training and testing sets
            random.shuffle(files)
            split_index = int(total_files * train_ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]
            
            # Make corresponding train and test subfolders
            train_subfolder = os.path.join(train_dir, subfolder)
            test_subfolder = os.path.join(test_dir, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)
            
            # Move files to train and test folders
            for file_name in train_files:
                shutil.move(os.path.join(subfolder_path, file_name), os.path.join(train_subfolder, file_name))
            for file_name in test_files:
                shutil.move(os.path.join(subfolder_path, file_name), os.path.join(test_subfolder, file_name))
                
    print("Data split completed.")




def show_random_images(data_dir):
    # Get list of class folders
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

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

# Example usage:
# show_random_images('path/to/your/dataset')



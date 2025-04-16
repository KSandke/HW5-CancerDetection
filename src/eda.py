# Exploratory Data Analysis for Histopathologic Cancer Detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import cv2 # OpenCV for image manipulation

# Configure settings (optional)
plt.style.use('ggplot')
sns.set_style('whitegrid')

# Define data paths
# Adjust the base path if your data is located elsewhere relative to the script
BASE_PATH = 'data/' # Corrected path relative to workspace root
TRAIN_DIR = os.path.join(BASE_PATH, 'train/')
TEST_DIR = os.path.join(BASE_PATH, 'test/')
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, 'train_labels.csv')

def load_data():
    """Loads the training labels.
    Returns:
        pandas.DataFrame: DataFrame containing image IDs and labels.
    """
    if not os.path.exists(TRAIN_LABELS_PATH):
        print(f"Error: Training labels file not found at {TRAIN_LABELS_PATH}")
        print("Please ensure the data has been downloaded and extracted correctly.")
        return None
    df_labels = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded labels for {len(df_labels)} images.")
    return df_labels

def explore_labels(df_labels):
    """Performs basic exploration of the labels.
    Args:
        df_labels (pandas.DataFrame): DataFrame with labels.
    """
    if df_labels is None:
        return

    print("\n--- Label Exploration ---")
    print(f"Shape of labels dataframe: {df_labels.shape}")
    print("\nFirst 5 entries:")
    print(df_labels.head())
    print("\nLabel distribution:")
    print(df_labels['label'].value_counts())
    print("\nPercentage distribution:")
    print(df_labels['label'].value_counts(normalize=True) * 100)

    # Plot distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df_labels)
    plt.title('Label Distribution (0 = No Cancer, 1 = Cancer)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    # Save the plot instead of showing interactively
    plot_filename = 'label_distribution.png'
    plt.savefig(plot_filename)
    print(f"\nSaved label distribution plot to {plot_filename}")
    plt.close() # Close the plot to free memory

def explore_images(df_labels, num_samples=5):
    """Loads and displays sample images from each class.
    Args:
        df_labels (pandas.DataFrame): DataFrame with labels.
        num_samples (int): Number of samples to show per class.
    """
    if df_labels is None:
        return

    print("\n--- Image Exploration ---")
    
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory not found at {TRAIN_DIR}")
        return
        
    positive_samples = df_labels[df_labels['label'] == 1].sample(num_samples)
    negative_samples = df_labels[df_labels['label'] == 0].sample(num_samples)

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    fig.suptitle('Sample Images', fontsize=16)

    for i, row in enumerate(positive_samples.itertuples()):
        img_path = os.path.join(TRAIN_DIR, f"{row.id}.tif")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"ID: {row.id[:6]}...\nLabel: {row.label} (Cancer)")
            axes[0, i].axis('off')
        else:
             axes[0, i].set_title(f"ID: {row.id[:6]}...\nNot Found")
             axes[0, i].axis('off')

    for i, row in enumerate(negative_samples.itertuples()):
        img_path = os.path.join(TRAIN_DIR, f"{row.id}.tif")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"ID: {row.id[:6]}...\nLabel: {row.label} (No Cancer)")
            axes[1, i].axis('off')
        else:
            axes[1, i].set_title(f"ID: {row.id[:6]}...\nNot Found")
            axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_filename = 'sample_images.png'
    plt.savefig(plot_filename)
    print(f"\nSaved sample images plot to {plot_filename}")
    plt.close()


if __name__ == "__main__":
    print("Starting EDA...")
    df = load_data()
    explore_labels(df)
    explore_images(df, num_samples=5)
    print("\nEDA script finished.")
    print(f"Check the generated plots: label_distribution.png and sample_images.png in {os.getcwd()}") 
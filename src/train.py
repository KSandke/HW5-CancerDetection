import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import cv2 # Ensure opencv-python is installed
from sklearn.model_selection import train_test_split
from PIL import Image # PIL is used by torchvision transforms
import time
import copy # For saving best model weights
from torch.cuda.amp import GradScaler, autocast # Import AMP components
import matplotlib.pyplot as plt # Ensure matplotlib is available
from torch.optim.lr_scheduler import CosineAnnealingLR # Import scheduler
from glob import glob # To find test files
from tqdm import tqdm # Progress bar for prediction

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# --- Configuration ---
# Data Paths (relative to the script's location, but typically run from workspace root)
BASE_PATH = 'data/' # Corrected path relative to workspace root
TRAIN_DIR = os.path.join(BASE_PATH, 'train/')
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, 'train_labels.csv')
MODEL_SAVE_PATH = 'models/' # Directory to save trained models (relative to workspace root)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Model & Training Parameters
MODEL_NAME = "resnet18" # Or "resnet34", etc.
IMAGE_SIZE = 96
BATCH_SIZE = 256  # Increased batch size
NUM_EPOCHS = 25  # Increased number of epochs
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 # Added weight decay
VALIDATION_SPLIT = 0.2 # 20% of training data for validation
RANDOM_STATE = 42 # For reproducible splits
NUM_WORKERS = 2 # Number of parallel workers for data loading (adjust based on system)
TEST_DIR = os.path.join(BASE_PATH, 'test/') # Define test directory
SUBMISSION_FILE = 'submission.csv' # Output filename

# Determine device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f'Memory Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    print(f'Memory Cached: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')


# --- Dataset Class ---
class CancerDataset(Dataset):
    """Custom Dataset for Histopathologic Cancer Detection"""
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Pandas dataframe with image ids and labels.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir,
                                f"{self.labels_frame.iloc[idx, 0]}.tif")
        # Load image using PIL, as torchvision transforms expect PIL images
        try:
            image = Image.open(img_name)
        except FileNotFoundError:
            print(f"Warning: File not found {img_name}")
            # Return a dummy image and label or handle appropriately
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), torch.tensor(0.0) 
            
        label = self.labels_frame.iloc[idx, 1]
        label = torch.tensor(float(label)) # Ensure label is float for BCEWithLogitsLoss

        if self.transform:
            image = self.transform(image)

        return image, label

# --- Transformations ---
# Define transformations for training and validation
# Normalization values for ImageNet pre-trained models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Augmentations for training data
train_transform = transforms.Compose([
    # transforms.Resize(IMAGE_SIZE), # Images are already 96x96
    # transforms.CenterCrop(IMAGE_SIZE), # Already 96x96
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    # Add more augmentations if needed (e.g., ColorJitter)
    transforms.ToTensor(),
    normalize,
])

# Only normalization for validation data
val_transform = transforms.Compose([
    # transforms.Resize(IMAGE_SIZE),
    # transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize,
])

# --- Data Loading ---
def prepare_data():
    print("Loading labels...")
    df_labels = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded {len(df_labels)} labels.")

    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(
        df_labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=df_labels['label'] # Important for imbalanced datasets
    )
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    print("Creating datasets...")
    train_dataset = CancerDataset(train_df, TRAIN_DIR, transform=train_transform)
    val_dataset = CancerDataset(val_df, TRAIN_DIR, transform=val_transform)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    return dataloaders, dataset_sizes

# --- Model Definition ---
def get_model(model_name=MODEL_NAME, num_classes=1, pretrained=True):
    print(f"Loading pre-trained model: {model_name}")
    model = None
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) # Modify final layer for binary classification
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    # Add other models (EfficientNet, etc.) here if needed
    else:
        raise ValueError(f"Model {model_name} not supported yet.")

    model = model.to(DEVICE)
    print("Model loaded successfully and moved to device.")
    return model

# --- Training Function ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # Initialize GradScaler for AMP, only if using CUDA
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f' Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}') # Print current LR
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            batch_num = 0
            num_batches = len(dataloaders[phase])
            for inputs, labels in dataloaders[phase]:
                batch_num += 1
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1) # Ensure labels are [batch_size, 1] and float for BCEWithLogitsLoss

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                # Use autocast for mixed precision
                with autocast(enabled=(DEVICE.type == 'cuda')):
                    outputs = model(inputs)
                    # Outputs are logits, preds are sigmoid probabilities > 0.5
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    # Scale loss and call backward
                    scaler.scale(loss).backward()
                    # Unscale gradients and step optimizer
                    scaler.step(optimizer)
                    # Update scaler for next iteration
                    scaler.update()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item()) # Use .item() to get scalar value
            else:
                 history['val_loss'].append(epoch_loss)
                 history['val_acc'].append(epoch_acc.item())


            # Deep copy the model if best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_best_val_acc.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best validation accuracy: {best_acc:.4f}. Model saved to {save_path}")
        
        # Step the scheduler after validation phase
        if scheduler:            
            scheduler.step()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# --- Plotting Function ---
def plot_history(history, num_epochs, model_name):
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_filename = f'training_history_{model_name}_e{num_epochs}.png'
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")
    plt.close() # Close the plot figure

# --- Test Dataset Class ---
class TestDataset(Dataset):
    """Dataset for loading test images without labels."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Find all .tif files in the test directory
        self.image_files = glob(os.path.join(self.image_dir, '*.tif'))
        if not self.image_files:
             print(f"Warning: No .tif files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_files[idx]
        # Extract image ID from filename
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}")
            # Return dummy image or handle error
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), "error_id"

        if self.transform:
            image = self.transform(image)

        return image, img_id

# --- Prediction Function ---
def generate_submission(model, test_dir, transform, batch_size, num_workers, device, submission_filename):
    print("\n--- Generating Test Predictions ---")
    
    # Load the best model weights
    best_model_path = os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_best_val_acc.pth')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model file not found at {best_model_path}")
        print("Please ensure the model was trained and saved correctly.")
        return
    print(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    print("Creating test dataset and dataloader...")
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    if len(test_dataset) == 0:
        print("Test dataset is empty. Cannot generate submission.")
        return

    predictions = []
    image_ids = []

    print("Running inference on test set...")
    with torch.no_grad(): # Disable gradient calculations
        for inputs, ids in tqdm(test_loader, desc="Predicting"): # Use tqdm for progress bar
            inputs = inputs.to(device)
            
            with autocast(enabled=(device.type == 'cuda')): # Use AMP for inference if available
                outputs = model(inputs)
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
            
            predictions.extend(probs.squeeze().cpu().tolist())
            image_ids.extend(ids)

    print(f"Generated {len(predictions)} predictions.")

    # Ensure correct number of predictions
    expected_test_images = 57458 # From Kaggle data page
    if len(predictions) != expected_test_images:
         print(f"Warning: Number of predictions ({len(predictions)}) does not match expected number of test images ({expected_test_images}).")
         # Check if TEST_DIR contains the correct images.

    # Create submission dataframe
    submission_df = pd.DataFrame({'id': image_ids, 'label': predictions})
    
    # Save submission file
    submission_df.to_csv(submission_filename, index=False)
    print(f"Submission file saved to {submission_filename}")

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Prepare Data
    dataloaders, dataset_sizes = prepare_data()

    # 2. Get Model
    model = get_model()

    # 3. Define Loss and Optimizer
    # Using BCEWithLogitsLoss which combines Sigmoid and BCELoss for better numerical stability
    criterion = nn.BCEWithLogitsLoss() 
    
    # Observe that all parameters are being optimized using AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Define the learning rate scheduler
    # Cosine Annealing decays LR over T_max epochs down to eta_min
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6) 

    # 4. Train Model
    trained_model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # 5. Save Final Model (optional, best is already saved during training)
    final_save_path = os.path.join(MODEL_SAVE_PATH, f'{MODEL_NAME}_final_epoch_{NUM_EPOCHS}.pth')
    torch.save(trained_model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

    # 6. Plot training history
    plot_history(history, NUM_EPOCHS, MODEL_NAME)

    # 7. Generate test set predictions and submission file
    # Use the validation transforms for the test set (no augmentation)
    generate_submission(
        model=trained_model, # Pass the model potentially holding best weights 
        test_dir=TEST_DIR,
        transform=val_transform, 
        batch_size=BATCH_SIZE * 2, # Can often use larger batch size for inference
        num_workers=NUM_WORKERS,
        device=DEVICE,
        submission_filename=SUBMISSION_FILE
    )

    print("\nTraining and Prediction script finished.") 
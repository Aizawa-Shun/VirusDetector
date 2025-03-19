"""
Utility functions related to data loading, pre-processing, transformations, etc.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

# Standard preprocessing transformations
def get_transforms(mode='eval'):
    """
    Get data transformation settings

    Parameters:
    -----------
    mode : str
        One of 'train', 'val' or 'eval'.

    Returns:
    --------
    transform : torchvision.transforms
        Transformations set to.
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # 'val' or 'eval'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    return transform


class VirusDataset(Dataset):
    """
    Virus image dataset
    """
    def __init__(self, image_paths, labels=None, transform=None):
        """
        Parameters:
        -----------
        image_paths : list
            List of image paths
        labels : list or None
            List of labels (inference mode if None)
        transform : torchvision.transforms or None
            Transformations to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.inference_mode = labels is None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.inference_mode:
            return image, img_path
        else:
            return image, self.labels[idx]


def prepare_train_val_data(data_dir, train_ratio=0.7):
    """
    Prepare data for training and validation

    Parameters:
    -----------
    data_dir : str
        Path of the data directory
    train_ratio : float
        Ratio of the training data.

    Returns:
    --------
    train_dataset : VirusDataset
        Training dataset
    val_dataset : VirusDataset
        Validation dataset
    """
    # Collect classes
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # Collect image paths and labels
    image_paths, labels = [], []
    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_folder):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(cls_folder, img_name))
                labels.append(class_to_idx[cls_name])
    
    # Split into training and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, train_size=train_ratio, stratify=labels, random_state=42
    )
    
    # Create datasets
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = VirusDataset(train_paths, train_labels, train_transform)
    val_dataset = VirusDataset(val_paths, val_labels, val_transform)
    
    return train_dataset, val_dataset


def prepare_eval_data(data_dir):
    """
    Prepare data for evaluation
    
    Parameters:
    -----------
    data_dir : str
        Directory containing images
        
    Returns:
    --------
    eval_dataset : VirusDataset
        Evaluation dataset
    image_paths : list
        List of image paths
    """
    image_paths = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(data_dir, filename))
    
    transform = get_transforms('eval')
    eval_dataset = VirusDataset(image_paths, labels=None, transform=transform)
    
    return eval_dataset, image_paths


def overlay_heatmap(img_path, heatmap, output_path=None, alpha=0.4):
    """
    Overlay heatmap on an image
    
    Parameters:
    -----------
    img_path : str
        Path to the original image
    heatmap : numpy.ndarray
        Heatmap
    output_path : str or None
        Output path (don't save if None)
    alpha : float
        Heatmap transparency (0-1)
        
    Returns:
    --------
    overlayed : numpy.ndarray
        Overlayed image
    """
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (img_resized.size[0], img_resized.size[1]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Combine image and heatmap
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
    
    return overlayed

def ensure_output_dirs(output_dir):
    """
    Create necessary output directories
    
    Parameters:
    -----------
    output_dir : str
        Path to the output directory
        
    Returns:
    --------
    output_dir : str
        Output directory path converted to absolute path
    """
    # Convert relative path to absolute path
    output_dir = os.path.abspath(output_dir)
    print(f"Creating output directories in: {output_dir}")
    
    # Create the output directory itself
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['sheet', 'image', 'graph', 'uncertainty']
    for subdir in subdirs:
        dir_path = os.path.join(output_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")
    
    return output_dir  # Return the converted absolute path
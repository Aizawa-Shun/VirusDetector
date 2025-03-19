"""
Utility modules providing common functions related to model creation, loading, evaluation, etc.
"""
import os
import torch
import torch.nn as nn
from torchvision import models

def create_virus_model(num_classes=3, pretrained=True):
    """
    Create a model for virus classification
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes (default: 3 classes [Positive, Negative, Uncertain])
    pretrained : bool
        Whether to use pre-trained weights
        
    Returns:
    --------
    model : torch.nn.Module
        ResNet-50 based model
    """
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    
    # Replace the output layer
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Softmax(dim=1)
    )
    
    return model

def load_model(model_path, device=None):
    """
    Load a trained model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    device : torch.device or None
        Device to load the model on (auto-detect if None)
        
    Returns:
    --------
    model : torch.nn.Module
        Loaded model
    target_layer : torch.nn.Module
        Target layer for Grad-CAM
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model
    model = create_virus_model()
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get the target layer for Grad-CAM
    target_layer = model.layer4[2].conv3
    
    return model, target_layer

def generate_gradcam(model, target_layer, img_tensor, class_idx):
    """
    Generate Grad-CAM
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    target_layer : torch.nn.Module
        Convolutional layer to focus on
    img_tensor : torch.Tensor
        Input image tensor (1, C, H, W)
    class_idx : int
        Target class index
        
    Returns:
    --------
    gradcam : numpy.ndarray
        Generated Grad-CAM
    """
    device = next(model.parameters()).device
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    
    # Backward pass
    model.zero_grad()
    target = output[:, class_idx].sum()
    target.backward()

    # Grad-CAM calculation
    gradient = gradients[0].squeeze()
    activation = activations[0].squeeze()
    weights = gradient.mean(dim=[1, 2])
    
    gradcam = torch.zeros(activation.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        gradcam += w * activation[i]
        
    gradcam = torch.relu(gradcam)
    gradcam = gradcam / (gradcam.max() + 1e-8)  # Prevent division by zero

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return gradcam.cpu().detach().numpy()

def apply_weights(scores, weights):
    """
    Apply weights to scores
    
    Parameters:
    -----------
    scores : torch.Tensor
        Model output scores
    weights : dict
        Weights for each class
        
    Returns:
    --------
    weighted_scores : torch.Tensor
        Weighted scores
    """
    weighted_scores = scores.clone()
    weighted_scores[:, 0] *= weights.get('Positive', 1.0)
    weighted_scores[:, 1] *= weights.get('Negative', 1.0)
    weighted_scores[:, 2] *= weights.get('Uncertain', 1.0)
    return weighted_scores

def calculate_weights():
    """
    Calculate weights for each class
    
    Returns:
    --------
    weights : dict
        Weights for each class
    """
    # Adjust as needed
    weights = {'Positive': 1.0, 'Negative': 1.0, 'Uncertain': 1.0}
    return weights

def extract_features(model, img_tensor):
    """
    Extract feature vectors from an image
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to use for feature extraction
    img_tensor : torch.Tensor
        Input image tensor
        
    Returns:
    --------
    feature_vector : numpy.ndarray
        Extracted feature vector
    """
    device = next(model.parameters()).device
    features = []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu().numpy())
        
    handle = model.layer4.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(img_tensor.to(device))
        
    handle.remove()
    
    # Convert feature maps to vector
    feature_vector = features[0].mean(axis=(2, 3)).reshape(-1)
    return feature_vector
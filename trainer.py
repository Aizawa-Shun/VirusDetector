"""
Module for model training and evaluation
"""
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model_utils import create_virus_model
from data_utils import prepare_train_val_data, ensure_output_dirs
from losses import get_loss_function
from visualization import plot_metrics

class EarlyStopping:
    """
    Early stopping implementation
    """
    def __init__(self, patience=5, delta=0):
        """
        Parameters:
        -----------
        patience : int
            Number of epochs to wait for improvement
        delta : float
            Minimum change to be considered as improvement
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Determine early stopping based on the latest validation loss
        
        Parameters:
        -----------
        val_loss : float
            Current validation loss
            
        Returns:
        --------
        early_stop : bool
            Whether to stop
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_model(config):
    """
    Execute model training
    
    Parameters:
    -----------
    config : dict
        Training configuration
    """
    # Extract configuration
    data_dir = config['data_dir']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    loss_function = config['loss_function']
    focal_alpha = config.get('focal_alpha', 0.25)
    focal_gamma = config.get('focal_gamma', 2.0)
    model_save_path = config['model_save_path']
    
    # Explicitly set output directory
    output_dir = config.get('output_dir')
    if not output_dir:
        # Use 'output' in the project root if output_dir is not in the config
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    
    print(f"Output directory set to: {output_dir}")
    
    # Create output directories - using ensure_output_dirs
    output_dir = ensure_output_dirs(output_dir)
    
    # Ensure output directory is explicitly set
    sheet_dir = os.path.join(output_dir, 'sheet')
    graph_dir = os.path.join(output_dir, 'graph')
    
    # Create output paths
    csv_path = os.path.join(sheet_dir, 'training_log.csv')
    loss_plot_path = os.path.join(graph_dir, 'loss_history.png')
    acc_plot_path = os.path.join(graph_dir, 'accuracy_history.png')
    
    # Debug output
    print(f"CSV path: {csv_path}")
    print(f"Loss plot path: {loss_plot_path}")
    print(f"Acc plot path: {acc_plot_path}")
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Prepare data
    train_dataset, val_dataset = prepare_train_val_data(data_dir)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Validation set size: {dataset_sizes['val']}")
    
    # Create model
    model = create_virus_model()
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = get_loss_function(loss_function, alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Initialize CSV logger
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Phase", "Loss", "Accuracy"])
    
    # Training history
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 40)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs[:, :2], 1)  # Exclude 'Uncertainty'
                    loss = criterion(outputs, labels)
                    
                    # Backward pass (training only)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Epoch average loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # CSV log
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, phase, epoch_loss, epoch_acc.item()])
            
            # Display results
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Update history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
        
        # Update scheduler
        scheduler.step()
        
        # Early stopping
        if early_stopping(val_loss_history[-1]):
            print("Early stopping triggered. Training stopped.")
            break
    
    print("Training complete.")
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot metrics - this is important
    print(f"Saving metrics plots to: {graph_dir}")
    
    # Explicitly specify paths
    plot_metrics(train_loss_history, val_loss_history, 'Loss', loss_plot_path)
    plot_metrics(train_acc_history, val_acc_history, 'Accuracy', acc_plot_path)
    
    return model
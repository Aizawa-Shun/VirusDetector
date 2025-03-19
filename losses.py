"""
Module related to loss functions
Includes implementation of special loss functions such as Focal Loss
"""
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation
    Reference: https://arxiv.org/abs/1708.02002
    
    A loss function that addresses class imbalance problems and 
    can focus more on difficult examples than easy ones.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Parameters:
        -----------
        alpha : float
            Weight coefficient to adjust class imbalance
        gamma : float
            Coefficient to adjust the balance between easy and hard examples
        reduction : str
            One of 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Parameters:
        -----------
        inputs : torch.Tensor
            Model outputs (N, C)
        targets : torch.Tensor
            Target classes (N,)
            
        Returns:
        --------
        loss : torch.Tensor
            Calculated loss
        """
        # Calculate cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate probability of target class
        pt = torch.exp(-ce_loss)
        
        # Apply Focal Loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(name, **kwargs):
    """
    Return a loss function with the specified name
    
    Parameters:
    -----------
    name : str
        Name of the loss function
    **kwargs : dict
        Additional arguments to pass to the loss function
        
    Returns:
    --------
    loss_fn : callable
        Loss function
    """
    if name.lower() == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif name.lower() == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function: {name}")
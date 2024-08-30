import numpy as np
import torch
import torch.nn as nn
from .dice import DiceLoss


def get_class_weights():
    class_pixels = np.array([497070029, 406090, 50003, 89182, 24286])
    total_pixels = np.sum(class_pixels)

    # Calculate weights as the inverse frequency
    class_weights = total_pixels / class_pixels

    # Normalize the weights, so they sum to 1
    normalized_weights = class_weights / np.sum(class_weights)

    # Convert weights to a PyTorch tensor
    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32)

    return weights_tensor


class WCEDiceLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, smooth=1e-5):
        super(WCEDiceLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.cross_entropy = nn.CrossEntropyLoss(weight=get_class_weights())
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        # Calculate Cross-Entropy Loss
        ce_loss = self.cross_entropy(pred, target)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Combine both losses with respective weights
        combined_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss)

        return combined_loss

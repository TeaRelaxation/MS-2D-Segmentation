import torch
import numpy as np
import segmentation_models_pytorch as smp
from .focal import FocalLoss
from .wce_dice import WCEDiceLoss
from .wce_dice_focal import WCEDiceFocalLoss


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


def select_loss(loss_name, device):
    class_weights = get_class_weights().to(device)
    if loss_name == "CE":
        return torch.nn.CrossEntropyLoss()
    if loss_name == "WCE":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "Dice":
        return smp.losses.DiceLoss(mode="multiclass")
    elif loss_name == "Focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_name == "WCEDice":
        return WCEDiceLoss(class_weights=class_weights)
    elif loss_name == "WCEDiceFocal":
        return WCEDiceFocalLoss(class_weights=class_weights)
    return None

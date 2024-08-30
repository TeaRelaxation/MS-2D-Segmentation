import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply softmax to the predicted logits to get probabilities
        # pred shape: (B,C,H,W)
        # target shape: (B,H,W)
        pred = torch.softmax(pred, dim=1)

        # Convert target to one-hot encoding and adjust dimensions
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # Compute the intersection and union
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        # Calculate Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss
        return 1 - dice.mean()

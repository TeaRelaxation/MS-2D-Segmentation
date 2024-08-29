import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply softmax to the predicted logits to get probabilities
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, weight_ce=0.3, weight_dice=0.7, smooth=1e-5):
        super(CEDiceLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        # Calculate Cross-Entropy Loss
        ce_loss = self.cross_entropy(pred, target)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Combine both losses with respective weights
        combined_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss)

        return combined_loss


class CEDiceFocalLoss(nn.Module):
    def __init__(self, weight_ce=0.2, weight_dice=0.5, weight_focal=0.3, focal_alpha=1, focal_gamma=2, smooth=1e-5):
        super(CEDiceFocalLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, pred, target):
        # Calculate Cross-Entropy Loss
        ce_loss = self.cross_entropy(pred, target)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Calculate Focal Loss
        focal_loss = self.focal_loss(pred, target)

        # Combine all losses with their respective weights
        combined_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss) + (self.weight_focal * focal_loss)

        return combined_loss

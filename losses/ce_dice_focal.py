import torch.nn as nn
from dice import DiceLoss
from focal import FocalLoss


class CEDiceFocalLoss(nn.Module):
    def __init__(self, weight_ce=0.2, weight_dice=0.6, weight_focal=0.2, focal_alpha=1, focal_gamma=2, smooth=1e-5):
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

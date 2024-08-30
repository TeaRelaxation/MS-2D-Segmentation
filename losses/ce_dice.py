import torch.nn as nn
from dice import DiceLoss


class CEDiceLoss(nn.Module):
    def __init__(self, weight_ce=0.2, weight_dice=0.8, smooth=1e-5):
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

import segmentation_models_pytorch as smp
import torch.nn as nn
from .focal import FocalLoss


class DiceFocalLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_focal=0.5):
        super(DiceFocalLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.focal_loss = FocalLoss(gamma=2.0)

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        combined_loss = (self.weight_dice * dice_loss) + (self.weight_focal * focal_loss)
        return combined_loss

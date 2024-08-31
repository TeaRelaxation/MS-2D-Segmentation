import segmentation_models_pytorch as smp
import torch.nn as nn
from .focal import FocalLoss


class WCEDiceFocalLoss(nn.Module):
    def __init__(self, class_weights, weight_ce=0.5, weight_dice=1.0, weight_focal=1.0):
        super(WCEDiceFocalLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.focal_loss = FocalLoss(gamma=2.0)

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        combined_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss) + (self.weight_focal * focal_loss)
        return combined_loss

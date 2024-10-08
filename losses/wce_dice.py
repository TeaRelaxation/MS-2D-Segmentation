import segmentation_models_pytorch as smp
import torch.nn as nn


class WCEDiceLoss(nn.Module):
    def __init__(self, class_weights, weight_ce=0.5, weight_dice=0.5):
        super(WCEDiceLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=-1)

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self.dice_loss(pred, target)
        combined_loss = (self.weight_ce * ce_loss) + (self.weight_dice * dice_loss)
        return combined_loss

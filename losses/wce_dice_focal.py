import segmentation_models_pytorch as smp
import torch.nn as nn
from .focal import FocalLoss


class WCEDiceFocalLoss(nn.Module):
    def __init__(self, class_weights, focal_class_weights, weight_ce=1.0, weight_dice=1.0, weight_focal=1.0):
        super(WCEDiceFocalLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=-1)
        self.focal_loss = FocalLoss(gamma=2.0, ignore_index=-1, alpha=focal_class_weights)

    def forward(self, pred, target):
        ce_loss = self.cross_entropy(pred, target)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)

        ce_loss_value = ce_loss.item()  # 2
        dice_loss_value = dice_loss.item()  # 4
        focal_loss_value = focal_loss.item()  # 8
        total_loss_value = ce_loss_value + dice_loss_value + focal_loss_value  # 14

        epsilon = 1e-8
        norm_ce_weight = total_loss_value / (ce_loss_value + epsilon)  # 7
        norm_dice_weight = total_loss_value / (dice_loss_value + epsilon)  # 3.5
        norm_focal_weight = total_loss_value / (focal_loss_value + epsilon)  # 1.75

        combined_loss = (self.weight_ce * norm_ce_weight * ce_loss) + \
                        (self.weight_dice * norm_dice_weight * dice_loss) + \
                        (self.weight_focal * norm_focal_weight * focal_loss)
        return combined_loss / 3

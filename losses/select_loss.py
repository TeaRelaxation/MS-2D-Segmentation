from torch import nn
from .dice import DiceLoss
from .focal import FocalLoss
from .wce_dice import WCEDiceLoss
from .ce_dice_focal import CEDiceFocalLoss


def select_loss(loss_name):
    if loss_name == "CE":
        return nn.CrossEntropyLoss
    elif loss_name == "Dice":
        return DiceLoss
    elif loss_name == "Focal":
        return FocalLoss
    elif loss_name == "WCEDice":
        return WCEDiceLoss
    elif loss_name == "CEDiceFocal":
        return CEDiceFocalLoss
    return None

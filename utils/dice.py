import torch


def dice_score(y_true, y_pred, num_classes):
    dice = 0
    smooth = 1e-5
    for i in range(num_classes):
        y_true_i = (y_true == i).float()
        y_pred_i = (y_pred == i).float()
        intersection = torch.sum(y_true_i * y_pred_i)
        dice += (2. * intersection + smooth) / (torch.sum(y_true_i) + torch.sum(y_pred_i) + smooth)
    return dice / num_classes

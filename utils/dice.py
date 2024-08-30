import torch.nn.functional as F


def dice_score(y_pred, y_true, num_classes):
    # Both pred and true shape:(B,H,W)
    smooth = 1e-5

    # Convert to one-hot encoding
    y_pred_one_hot = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float()
    y_true_one_hot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()

    # Compute the intersection and union
    intersection = (y_pred_one_hot * y_true_one_hot).sum(dim=(2, 3))
    union = y_pred_one_hot.sum(dim=(2, 3)) + y_true_one_hot.sum(dim=(2, 3))

    # Calculate Dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()

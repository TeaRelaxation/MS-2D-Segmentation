import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def get_resizer(height, width, resize_type):
    if resize_type == "resize":
        return A.Resize(height=height, width=width, p=1.0)
    elif resize_type == "pad":
        return A.PadIfNeeded(
            min_height=height,
            min_width=width,
            border_mode=0,
            value=0,
            mask_value=0,
            p=1.0
        )
    return None


def get_augmentor(mode, height, width, resize_type):
    if mode == "train":
        return A.Compose([
            get_resizer(height, width, resize_type),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            ToTensorV2(),  # HWC -> CHW (Only image not mask)
        ])
    elif mode == "test":
        return A.Compose([
            get_resizer(height, width, resize_type),
            ToTensorV2(),
        ])
    return None


def get_normalizer(mean, std):
    return transforms.Normalize(mean=[mean], std=[std])

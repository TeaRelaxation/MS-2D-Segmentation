import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def get_augmentor(mode):
    if mode == "train":
        return A.Compose([
            A.Resize(height=224, width=192, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            ToTensorV2(),  # HWC -> CHW (Only image not mask)
        ])
    elif mode == "test":
        return A.Compose([
            A.Resize(height=224, width=192, p=1.0),
            ToTensorV2(),
        ])
    return None


def get_normalizer(mean, std):
    return transforms.Normalize(mean=[mean], std=[std])

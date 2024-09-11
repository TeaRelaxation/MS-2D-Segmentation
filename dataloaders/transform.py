import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def get_padding(height, width):
    return A.PadIfNeeded(
        min_height=height,
        min_width=width,
        border_mode=0,
        value=0,
        mask_value=-1,
        p=1.0
    )


def get_augmentor(mode, height, width):
    if mode == "train":
        return A.Compose([
            get_padding(height, width),
            A.RandomCrop(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            # A.OneOf([
            #     # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            #     A.GridDistortion(p=0.5),
            #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            # ], p=0.8),
            # # A.CLAHE(p=0.8),
            # A.RandomBrightnessContrast(p=0.8),
            # A.RandomGamma(p=0.8),
            ToTensorV2()  # HWC -> CHW (Only image not mask)
        ])
    elif mode == "test":
        return A.Compose([
            get_padding(height, width),
            ToTensorV2(),
        ])
    return None


def get_normalizer(mean, std):
    return transforms.Normalize(mean=[mean], std=[std])

from .ms import MSDataset
from .brats import BraTSDataset
from .transform import get_augmentor, get_normalizer


def select_data(
        dataset_name,
        dataset_path,
        max_pixel,
        mean,
        std,
        crop_h,
        crop_w,
        infer_h,
        infer_w,
        in_channels,
        is_imagenet
):
    train_data = None
    val_data = None

    train_augmentor = get_augmentor("train", crop_h, crop_w)
    test_augmentor = get_augmentor("test", infer_h, infer_w)

    if in_channels == 3:
        if is_imagenet == "True":
            normalizer = get_normalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalizer = get_normalizer(mean=[mean, mean, mean], std=[std, std, std])
    else:
        if is_imagenet == "True":
            normalizer = get_normalizer(mean=[mean], std=[std])
        else:
            normalizer = get_normalizer(mean=[mean], std=[std])

    if dataset_name == "MS":
        train_data = MSDataset(
            root_dir=f"{dataset_path}/train",
            augmentor=train_augmentor,
            max_pixel=max_pixel,
            normalizer=normalizer,
            in_channels=in_channels
        )
        val_data = MSDataset(
            root_dir=f"{dataset_path}/test",
            augmentor=test_augmentor,
            max_pixel=max_pixel,
            normalizer=normalizer,
            in_channels=in_channels
        )
    elif dataset_name == "BraTS":
        train_data = BraTSDataset(
            root_dir=f"{dataset_path}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
            split="train",
            augmentor=train_augmentor,
            max_pixel=max_pixel,
            normalizer=normalizer,
            in_channels=in_channels
        )
        val_data = BraTSDataset(
            root_dir=f"{dataset_path}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
            split="val",
            augmentor=test_augmentor,
            max_pixel=max_pixel,
            normalizer=normalizer,
            in_channels=in_channels
        )
    return train_data, val_data

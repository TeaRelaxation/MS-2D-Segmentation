from .ms import MSDataset
from .transform import get_augmentor, get_normalizer


def select_data(dataset_name, dataset_path, height, width, resize_type):
    train_data = None
    val_data = None
    train_augmentor = get_augmentor("train", height, width, resize_type)
    test_augmentor = get_augmentor("test", height, width, resize_type)
    if dataset_name == "MS":
        normalizer = get_normalizer(mean=47.532, std=51.077)
        train_data = MSDataset(root_dir=f"{dataset_path}/train", augmentor=train_augmentor, normalizer=normalizer)
        val_data = MSDataset(root_dir=f"{dataset_path}/test", augmentor=test_augmentor, normalizer=normalizer)
    return train_data, val_data

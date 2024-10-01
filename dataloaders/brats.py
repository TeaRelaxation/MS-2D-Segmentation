import os
import random
import nibabel as nib
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, root_dir, split, augmentor, max_pixel, normalizer, in_channels):
        self.root_dir = root_dir
        self.split = split
        self.augmentor = augmentor
        self.max_pixel = max_pixel
        self.normalizer = normalizer
        self.in_channels = in_channels

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Loop over all subdirectories
        for subdir in os.listdir(self.root_dir):  # subdir: BraTS20_Training_001
            flair_filename = f"{subdir}_flair.nii"
            seg_filename = f"{subdir}_seg.nii"
            flair_path = os.path.join(self.root_dir, subdir, flair_filename)
            seg_path = os.path.join(self.root_dir, subdir, seg_filename)
            if os.path.exists(flair_path) and os.path.exists(seg_path):
                samples.append((flair_path, seg_path))

        # Split train val
        random.seed(42)
        random.shuffle(samples)
        split_index = int(0.8 * len(samples))

        if self.split == "train":
            return samples[:split_index]  # 80%
        elif self.split == "val":
            return samples[split_index:]  # 20%
        return None

    def __len__(self):
        return len(self.samples) * 155  # Assuming the z-dimension is 155

    def __getitem__(self, idx):
        volume_idx = idx // 155
        slice_idx = idx % 155

        flair_path, lesion_path = self.samples[volume_idx]

        flair_img = nib.load(flair_path).get_fdata()
        lesion_img = nib.load(lesion_path).get_fdata()

        # Scale to [0, 1)
        if self.max_pixel == 0.0:
            flair_img /= flair_img.max()
        else:
            flair_img /= self.max_pixel

        # Rename label 4 to 3
        lesion_img[lesion_img == 4] = 3

        flair_slice = flair_img[:, :, slice_idx]
        lesion_slice = lesion_img[:, :, slice_idx]

        flair_slice = flair_slice[None, :, :].astype("float32")  # (C=1,W=240,H=240)
        lesion_slice = lesion_slice.astype("float32")  # (W=240,H=240)

        flair_slice = flair_slice.transpose(2, 1, 0)  # (H=240,W=240,C=1)
        lesion_slice = lesion_slice.transpose(1, 0)  # (H=240,W=240)

        augmented = self.augmentor(image=flair_slice, mask=lesion_slice)
        flair_slice = augmented["image"]  # (C=1,H=240,W=240)
        lesion_slice = augmented["mask"]  # (H=240,W=240)

        if self.in_channels == 3:
            flair_slice = flair_slice.repeat(3, 1, 1)  # Repeat the channel dimension 3 times

        flair_slice = self.normalizer(flair_slice)  # (C=1or3,H=240,W=240)

        return flair_slice, lesion_slice

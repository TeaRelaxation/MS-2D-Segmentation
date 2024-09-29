import os
import nibabel as nib
from torch.utils.data import Dataset


class MSDataset(Dataset):
    def __init__(self, root_dir, augmentor, max_pixel, normalizer, in_channels):
        self.root_dir = root_dir
        self.augmentor = augmentor
        self.max_pixel = max_pixel
        self.normalizer = normalizer
        self.in_channels = in_channels

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # Loop over all subdirectories
        for subdir in sorted(os.listdir(self.root_dir)):
            file_index = int(subdir)
            flair_filename = f"{file_index}_rr_mni_flair.nii"
            lesion_filename = f"{file_index}_rr_mni_lesion.nii"
            flair_path = os.path.join(self.root_dir, subdir, flair_filename)
            lesion_path = os.path.join(self.root_dir, subdir, lesion_filename)
            if os.path.exists(flair_path) and os.path.exists(lesion_path):
                samples.append((flair_path, lesion_path))
        return samples

    def __len__(self):
        return len(self.samples) * 181  # Assuming the z-dimension is 181

    def __getitem__(self, idx):
        volume_idx = idx // 181
        slice_idx = idx % 181

        flair_path, lesion_path = self.samples[volume_idx]

        flair_img = nib.load(flair_path).get_fdata()
        lesion_img = nib.load(lesion_path).get_fdata()

        flair_slice = flair_img[:, :, slice_idx]
        lesion_slice = lesion_img[:, :, slice_idx]

        flair_slice = flair_slice[None, :, :].astype("float32")  # (C=1,W=181,H=217)
        lesion_slice = lesion_slice.astype("float32")  # (W=181,H=217)

        flair_slice = flair_slice.transpose(2, 1, 0)  # (H=217,W=181,C=1)
        lesion_slice = lesion_slice.transpose(1, 0)  # (H=217,W=181)

        # Scale to [0, 1)
        flair_slice /= self.max_pixel

        augmented = self.augmentor(image=flair_slice, mask=lesion_slice)
        flair_slice = augmented["image"]  # (C=1,H=217,W=181)
        lesion_slice = augmented["mask"]  # (H=217,W=181)

        if self.in_channels == 3:
            flair_slice = flair_slice.repeat(3, 1, 1)  # Repeat the channel dimension 3 times

        flair_slice = self.normalizer(flair_slice)  # (C=1or3,H=217,W=181)

        return flair_slice, lesion_slice

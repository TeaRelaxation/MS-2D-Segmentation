import os
import nibabel as nib
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from dataloaders.ms import MSDataset
from dataloaders.transform import get_augmentor, get_normalizer
from utils.trainer import remove_pad, convert_3d

in_channels = 1
n_classes = 5
max_pixel = 375.3621
mean = 0.449
std = 0.226
height = 224
width = 192
depth = 181
batch_size = 32
dataset_path = "../datasets/MS/"
model_path = "../playground/best_model.pth"
save_dir = "../playground/outputs/val_inference"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

augmentor = get_augmentor("test", height, width)
normalizer = get_normalizer(mean=[mean], std=[std])
data = MSDataset(
    root_dir=f"{dataset_path}/test",
    augmentor=augmentor,
    max_pixel=max_pixel,
    normalizer=normalizer,
    in_channels=in_channels
)
dataloader = DataLoader(data, shuffle=False, batch_size=batch_size)

model = smp.UnetPlusPlus(
    encoder_name="timm-resnest269e",
    in_channels=in_channels,
    classes=n_classes,
    encoder_weights=None
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

preds_list = []
targets_list = []
with torch.no_grad():
    for flair_slice, lesion_slice in dataloader:
        flair_slice = flair_slice.to(device)
        lesion_slice = lesion_slice.to(device)

        output = model(flair_slice)
        predicted_labels = torch.argmax(output, dim=1).long()
        preds_list.append(predicted_labels)

        lesion_slice = lesion_slice.long()
        targets_list.append(lesion_slice)

preds_tensor = torch.cat(preds_list, dim=0)
targets_tensor = torch.cat(targets_list, dim=0)

preds_cropped_tensor, targets_cropped_tensor = remove_pad(preds_tensor, targets_tensor)

preds_3d = convert_3d(preds_cropped_tensor, depth)  # (N,H,W,D)
targets_3d = convert_3d(targets_cropped_tensor, depth)

os.makedirs(save_dir, exist_ok=True)

for i in range(preds_3d.shape[0]):
    # Convert tensor to numpy uint8
    sample = preds_3d[i].cpu().numpy().astype("uint8")

    # Permute the sample from (H, W, D) to (W, H, D)
    sample = np.transpose(sample, (1, 0, 2))

    # Convert the permuted sample to a NIfTI image
    nifti_image = nib.Nifti1Image(sample, affine=np.eye(4))

    # Save the NIfTI file (you can use a unique name for each file)
    nib.save(nifti_image, f"{save_dir}/inference_{i+1}.nii")

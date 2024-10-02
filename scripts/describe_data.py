import os
import nibabel as nib
import numpy as np

images = []
masks = []

# Loop over all subdirectories
root_dir = "../datasets/MS/test"

for subdir in sorted(os.listdir(root_dir)):
    file_index = int(subdir)
    flair_filename = f"{file_index}_rr_mni_flair.nii"
    lesion_filename = f"{file_index}_rr_mni_lesion.nii"
    flair_path = os.path.join(root_dir, subdir, flair_filename)
    lesion_path = os.path.join(root_dir, subdir, lesion_filename)

    flair_img = nib.load(flair_path).get_fdata()
    lesion_img = nib.load(lesion_path).get_fdata()

    images.append(flair_img)
    masks.append(lesion_img)

# Convert the lists to NumPy arrays
images_array = np.array(images)
masks_array = np.array(masks)

# Optional: print shapes to confirm
print(f"Images array shape: {images_array.shape}")
print(f"Masks array shape: {masks_array.shape}")

# Calculate statistics for images
images_max = np.max(images_array)
images_min = np.min(images_array)
images_mean = np.mean(images_array)
images_std = np.std(images_array)

# Calculate statistics for masks
masks_max = np.max(masks_array)
masks_min = np.min(masks_array)
masks_mean = np.mean(masks_array)
masks_std = np.std(masks_array)

# Count the number of pixels for each class in masks
class_0_pixels = np.sum(masks_array == 0)
class_1_pixels = np.sum(masks_array == 1)
class_2_pixels = np.sum(masks_array == 2)
class_3_pixels = np.sum(masks_array == 3)
class_4_pixels = np.sum(masks_array == 4)

# Print the calculated statistics
print("Images statistics:")
print(f"Max: {images_max}")
print(f"Min: {images_min}")
print(f"Mean: {images_mean}")
print(f"Std: {images_std}")

print("\nMasks statistics:")
print(f"Max: {masks_max}")
print(f"Min: {masks_min}")
print(f"Mean: {masks_mean}")
print(f"Std: {masks_std}")

# Print the number of pixels for each class
print("\nNumber of pixels in each class:")
print(f"Class 0: {class_0_pixels}")
print(f"Class 1: {class_1_pixels}")
print(f"Class 2: {class_2_pixels}")
print(f"Class 3: {class_3_pixels}")
print(f"Class 4: {class_4_pixels}")

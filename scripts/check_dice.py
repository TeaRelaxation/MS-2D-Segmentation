import nibabel as nib
import torch
from segmentation_models_pytorch.metrics import functional as metric

preds_list = []
targets_list = []

j = 10

for i in range(1, 2):
    pred = nib.load(f"../playground/outputs/val_inference/inference_{i}.nii").get_fdata()  # (W,H)
    target = nib.load(f"../playground/outputs/val_target/target_{i}.nii").get_fdata()

    pred = torch.tensor(pred, dtype=torch.float32).long()
    target = torch.tensor(target, dtype=torch.float32).long()

    preds_list.append(pred)
    targets_list.append(target)

preds_tensor = torch.stack(preds_list)
targets_tensor = torch.stack(targets_list)

# (N,d1,d2,d3,...) -> (N,d1*d2*d3*...)
output = preds_tensor.reshape(preds_tensor.shape[0], -1)
label = targets_tensor.reshape(targets_tensor.shape[0], -1)

tp, fp, fn, tn = metric.get_stats(
    output,
    label,
    mode="multiclass",
    ignore_index=-1,
    num_classes=5
)

dice_score = metric.f1_score(tp, fp, fn, tn, reduction="none").mean(dim=0)
avg_all_classes = dice_score.mean().item()
avg_no_class0 = dice_score[1:].mean().item()

print(dice_score)
print(avg_all_classes)
print(avg_no_class0)

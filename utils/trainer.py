import torch
from segmentation_models_pytorch.metrics import functional as metric
from torch.utils.data import DataLoader
from .metrics import Metrics


class Trainer:
    def __init__(
            self,
            train_dataset,
            val_dataset,
            model,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            batch_size,
            device,
            logger,
            n_classes,
            workers
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.logger = logger
        self.n_classes = n_classes

        self.model = model.to(self.device)
        common_loader_params = {'batch_size': batch_size, 'pin_memory': True, 'num_workers': workers}
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
        self.val_dataloader = DataLoader(val_dataset, shuffle=False, **common_loader_params)
        self.best_dice_score = -float('inf')

        self.train_metrics = Metrics(reduction="macro-imagewise", n_classes=self.n_classes)
        self.val_metrics = Metrics(reduction="macro-imagewise", n_classes=self.n_classes)
        self.history = {}

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()

            for flair_slice, lesion_slice in self.train_dataloader:
                flair_slice = flair_slice.to(self.device)
                lesion_slice = lesion_slice.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(flair_slice)

                # Convert lesion_slice to long type as required by CrossEntropyLoss
                lesion_slice = lesion_slice.long()

                loss = self.criterion(output, lesion_slice)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    predicted_labels = torch.argmax(output, dim=1).long()
                    self.train_metrics.iteration_end(output=predicted_labels, label=lesion_slice, loss=loss)

            self.scheduler.step()

            n_batches = len(self.train_dataloader)
            self.train_metrics.epoch_end(n_batches)

            # Evaluate on validation set
            preds_list, targets_list = self.evaluate()

            # Log and print results
            self.logger.log_train(
                epochs=self.num_epochs,
                train_metrics=self.train_metrics,
                val_metrics=self.val_metrics
            )

            # Save the model if it has the best Dice score
            current_val_dice = self.val_metrics.dice_list[-1]
            if current_val_dice > self.best_dice_score:
                self.best_dice_score = current_val_dice
                self.logger.save_model(self.model, "best_model.pth")

            # 3D Evaluation
            self.evaluate_3d(preds_list, targets_list)

            self.logger.print_log("-" * 50)

        # Save the model from the last epoch
        self.logger.save_model(self.model, "last_model.pth")

        self.history = {
            "train": self.train_metrics.get_history_dict(),
            "val": self.val_metrics.get_history_dict()
        }
        self.logger.save_history(self.history, "history.pkl")

    def evaluate(self):
        self.model.eval()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for flair_slice, lesion_slice in self.val_dataloader:
                flair_slice = flair_slice.to(self.device)
                lesion_slice = lesion_slice.to(self.device)

                output = self.model(flair_slice)

                # Convert lesion_slice to long type as required by CrossEntropyLoss
                lesion_slice = lesion_slice.long()
                targets_list.append(lesion_slice)

                # Calculate loss
                loss = self.criterion(output, lesion_slice)

                predicted_labels = torch.argmax(output, dim=1).long()
                preds_list.append(predicted_labels)

                self.val_metrics.iteration_end(output=predicted_labels, label=lesion_slice, loss=loss)

            n_batches = len(self.val_dataloader)
            self.val_metrics.epoch_end(n_batches)

        return preds_list, targets_list

    def evaluate_3d(self, preds_list, targets_list):
        # convert list to tensor: (N,H,W)
        preds_tensor = torch.cat(preds_list, dim=0)
        targets_tensor = torch.cat(targets_list, dim=0)

        # Remove padding
        preds_cropped_tensor, targets_cropped_tensor = remove_pad(preds_tensor, targets_tensor)

        # Convert 2D to 3D
        split_size = 181
        preds_3d = convert_3d(preds_cropped_tensor, split_size)
        targets_3d = convert_3d(targets_cropped_tensor, split_size)

        # (N,H,W,D) -> (N,H*W*D)
        preds_flat = preds_3d.reshape(preds_3d.shape[0], -1)
        targets_flat = targets_3d.reshape(targets_3d.shape[0], -1)

        tp, fp, fn, tn = metric.get_stats(
            preds_flat,
            targets_flat,
            mode="multiclass",
            ignore_index=-1,
            num_classes=self.n_classes
        )

        dice_3d = metric.f1_score(tp, fp, fn, tn, reduction="macro-imagewise").item()
        print(f"3D Dice: {dice_3d}")
        # print(f"3D Dice class 0: {dice_3d[0]}")
        # print(f"3D Dice class 1: {dice_3d[1]}")
        # print(f"3D Dice class 2: {dice_3d[2]}")
        # print(f"3D Dice class 3: {dice_3d[3]}")
        # print(f"3D Dice class 4: {dice_3d[4]}")


def remove_pad(preds, targets):
    # Find where padded values are not -1
    mask = targets != -1

    # Find indices where values are not -1 along height and width
    non_padded_indices = torch.where(mask)

    # Get the bounding box for cropping
    min_height, max_height = torch.min(non_padded_indices[1]), torch.max(non_padded_indices[1])
    min_width, max_width = torch.min(non_padded_indices[2]), torch.max(non_padded_indices[2])

    # Crop the tensor by slicing along the height and width
    preds_cropped_tensor = preds[:, min_height:max_height + 1, min_width:max_width + 1]
    targets_cropped_tensor = targets[:, min_height:max_height + 1, min_width:max_width + 1]

    return preds_cropped_tensor, targets_cropped_tensor


def convert_3d(tensor_2d, split_size):
    height = tensor_2d.size(1)
    width = tensor_2d.size(2)
    num_splits = tensor_2d.size(0) // split_size
    tensor_3d = tensor_2d[:num_splits * split_size].reshape(num_splits, split_size, height, width)
    tensor_3d = tensor_3d.permute(0, 2, 3, 1)
    return tensor_3d

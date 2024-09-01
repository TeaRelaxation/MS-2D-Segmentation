import logging
import torch
from segmentation_models_pytorch.metrics import functional as metric
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self,
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
                 workers):
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

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            train_dice_score = 0

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

                epoch_loss += loss.item()

                with torch.no_grad():
                    predicted_labels = torch.argmax(output, dim=1).long()

                    tp, fp, fn, tn = metric.get_stats(
                        predicted_labels,
                        lesion_slice,
                        mode="multiclass",
                        ignore_index=-1,
                        num_classes=self.n_classes
                    )

                    train_dice_score += metric.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")

                # print("End of batch")  # Debug log

            self.scheduler.step()

            epoch_loss /= len(self.train_dataloader)
            train_dice_score /= len(self.train_dataloader)

            # Evaluate on validation set
            val_loss, val_dice_score = self.evaluate()

            # Log and print results
            log_text = f"Epoch {epoch + 1}/{self.num_epochs} --- " \
                       f"Train Loss: {epoch_loss:.4f} " \
                       f"Train Dice: {train_dice_score:.4f} " \
                       f"Val Loss: {val_loss:.4f} " \
                       f"Val Dice: {val_dice_score:.4f}"

            logging.info(log_text)
            print(log_text)

            # Save the model if it has the best Dice score
            if val_dice_score > self.best_dice_score:
                self.best_dice_score = val_dice_score
                self.logger.save_model(self.model, "best_model.pth")

        # Save the model from the last epoch
        self.logger.save_model(self.model, "last_model.pth")

    def evaluate(self):
        self.model.eval()
        val_dice_score = 0
        val_loss = 0

        with torch.no_grad():
            for flair_slice, lesion_slice in self.val_dataloader:
                flair_slice = flair_slice.to(self.device)
                lesion_slice = lesion_slice.to(self.device)

                output = self.model(flair_slice)

                # Convert lesion_slice to long type as required by CrossEntropyLoss
                lesion_slice = lesion_slice.long()

                # Calculate loss
                loss = self.criterion(output, lesion_slice)
                val_loss += loss.item()

                predicted_labels = torch.argmax(output, dim=1).long()

                tp, fp, fn, tn = metric.get_stats(
                    predicted_labels,
                    lesion_slice,
                    mode="multiclass",
                    ignore_index=-1,
                    num_classes=self.n_classes
                )

                val_dice_score += metric.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")

        # Average the validation loss and Dice score over all batches
        val_loss /= len(self.val_dataloader)
        val_dice_score /= len(self.val_dataloader)

        return val_loss, val_dice_score

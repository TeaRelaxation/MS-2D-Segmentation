import torch
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
                    self.train_metrics.iteration_end(output=output, label=lesion_slice, loss=loss)

            self.scheduler.step()

            n_batches = len(self.train_dataloader)
            self.train_metrics.epoch_end(n_batches)

            # Evaluate on validation set
            self.evaluate()

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

        with torch.no_grad():
            for flair_slice, lesion_slice in self.val_dataloader:
                flair_slice = flair_slice.to(self.device)
                lesion_slice = lesion_slice.to(self.device)

                output = self.model(flair_slice)

                # Convert lesion_slice to long type as required by CrossEntropyLoss
                lesion_slice = lesion_slice.long()

                # Calculate loss
                loss = self.criterion(output, lesion_slice)

                self.val_metrics.iteration_end(output=output, label=lesion_slice, loss=loss)

            n_batches = len(self.val_dataloader)
            self.val_metrics.epoch_end(n_batches)

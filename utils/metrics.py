from segmentation_models_pytorch.metrics import functional as metric
import torch


class Metrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes

        self.metric_names = [
            "loss", "dice", "accuracy", "sensitivity",
            "specificity", "precision", "recall"
        ]

        # Initialize dictionaries to store epoch metrics and history for each class
        self.epoch_metrics = {name: torch.zeros(n_classes) for name in self.metric_names}
        self.history = {name: [] for name in self.metric_names}

    def iteration_end(self, output, label, loss):
        # (N,d1,d2,d3,...) -> (N,d1*d2*d3*...)
        output = output.reshape(output.shape[0], -1)
        label = label.reshape(label.shape[0], -1)

        tp, fp, fn, tn = metric.get_stats(
            output,
            label,
            mode="multiclass",
            ignore_index=-1,
            num_classes=self.n_classes
        )

        # Calculate metrics without reduction (returns tensor of shape (N, C))
        metric_functions = {
            "loss": lambda: loss.item(),  # loss is already a scalar
            "dice": lambda: metric.f1_score(tp, fp, fn, tn, reduction="none"),
            "accuracy": lambda: metric.accuracy(tp, fp, fn, tn, reduction="none"),
            "sensitivity": lambda: metric.sensitivity(tp, fp, fn, tn, reduction="none"),
            "specificity": lambda: metric.specificity(tp, fp, fn, tn, reduction="none"),
            "precision": lambda: metric.precision(tp, fp, fn, tn, reduction="none"),
            "recall": lambda: metric.recall(tp, fp, fn, tn, reduction="none"),
        }

        # Update metrics by averaging over batch size (dim=0)
        for name in self.metric_names:
            if name == "loss":
                self.epoch_metrics[name] += metric_functions[name]()
            else:
                self.epoch_metrics[name] += metric_functions[name]().mean(dim=0)  # Mean across batch dimension (N)

    def epoch_end(self, n_batches):
        # Average the metrics over the number of batches
        for name in self.metric_names:
            self.epoch_metrics[name] /= n_batches

            # Compute average over all classes
            avg_all_classes = self.epoch_metrics[name].mean().item()

            # Compute average over classes 1, 2, 3, ..., excluding class 0
            avg_no_class0 = self.epoch_metrics[name][1:].mean().item()

            # Append the tuple of per-class scores + the two averages
            metric_tuple = (*self.epoch_metrics[name].tolist(), avg_all_classes, avg_no_class0)
            self.history[name].append(metric_tuple)

        # Reset epoch metrics for the next epoch
        self.reset_epoch_values()

    def reset_epoch_values(self):
        # Reset the epoch metrics (zero tensor for each class)
        self.epoch_metrics = {name: torch.zeros(self.n_classes) for name in self.metric_names}

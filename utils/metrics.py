from segmentation_models_pytorch.metrics import functional as metric


class Metrics:
    def __init__(self, reduction, n_classes):
        self.reduction = reduction
        self.n_classes = n_classes

        self.epoch_loss = 0
        self.epoch_dice = 0
        self.epoch_accuracy = 0
        self.epoch_sensitivity = 0
        self.epoch_specificity = 0
        self.epoch_precision = 0
        self.epoch_recall = 0

        self.loss_list = []
        self.dice_list = []
        self.accuracy_list = []
        self.sensitivity_list = []
        self.specificity_list = []
        self.precision_list = []
        self.recall_list = []

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

        self.epoch_loss += loss.item()
        self.epoch_dice += metric.f1_score(tp, fp, fn, tn, reduction=self.reduction).item()
        self.epoch_accuracy += metric.accuracy(tp, fp, fn, tn, reduction=self.reduction).item()
        self.epoch_sensitivity += metric.sensitivity(tp, fp, fn, tn, reduction=self.reduction).item()
        self.epoch_specificity += metric.specificity(tp, fp, fn, tn, reduction=self.reduction).item()
        self.epoch_precision += metric.precision(tp, fp, fn, tn, reduction=self.reduction).item()
        self.epoch_recall += metric.recall(tp, fp, fn, tn, reduction=self.reduction).item()

    def epoch_end(self, n_batches):
        self.epoch_loss /= n_batches
        self.epoch_dice /= n_batches
        self.epoch_accuracy /= n_batches
        self.epoch_sensitivity /= n_batches
        self.epoch_specificity /= n_batches
        self.epoch_precision /= n_batches
        self.epoch_recall /= n_batches

        self.loss_list.append(self.epoch_loss)
        self.dice_list.append(self.epoch_dice)
        self.accuracy_list.append(self.epoch_accuracy)
        self.sensitivity_list.append(self.epoch_sensitivity)
        self.specificity_list.append(self.epoch_specificity)
        self.precision_list.append(self.epoch_precision)
        self.recall_list.append(self.epoch_recall)

        self.reset_epoch_values()

    def get_history_dict(self):
        return {
            "loss": self.loss_list,
            "dice": self.dice_list,
            "accuracy": self.accuracy_list,
            "sensitivity": self.sensitivity_list,
            "specificity": self.specificity_list,
            "precision": self.precision_list,
            "recall": self.recall_list
        }

    def reset_epoch_values(self):
        self.epoch_loss = 0
        self.epoch_dice = 0
        self.epoch_accuracy = 0
        self.epoch_sensitivity = 0
        self.epoch_specificity = 0
        self.epoch_precision = 0
        self.epoch_recall = 0

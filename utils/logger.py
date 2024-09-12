import logging
import os
import pickle
import torch


class Logger:
    def __init__(self, root_dir, experiment_name):
        self.path = os.path.join(root_dir, experiment_name)
        os.makedirs(self.path, exist_ok=True)
        log_path = os.path.abspath(os.path.join(self.path, "logs.log"))

        formatter = logging.Formatter(fmt='%(asctime)s %(message)s')
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)

        self.logger = logging.getLogger("root")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(fh)

    def save_model(self, model, filename):
        model_path = os.path.join(self.path, filename)
        torch.save(model.state_dict(), model_path)
        self.print_log(f"Model saved {filename}")

    def save_history(self, history, filename):
        history_path = os.path.join(self.path, filename)
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        self.print_log(f"History saved {filename}")

    def log_train(self, epochs, train_metrics, val_metrics, val_3d_metrics):
        epoch = len(train_metrics.loss_list)
        log_text = (
            f"Epoch {epoch}/{epochs}\n"
            f"Train Loss: {train_metrics.loss_list[-1]:.4f}\n"
            f"Train Dice: {train_metrics.dice_list[-1]:.4f}\n"
            f"Train Accu: {train_metrics.accuracy_list[-1]:.4f}\n"
            f"Train Sens: {train_metrics.sensitivity_list[-1]:.4f}\n"
            f"Train Spec: {train_metrics.specificity_list[-1]:.4f}\n"
            f"Train Prec: {train_metrics.precision_list[-1]:.4f}\n"
            f"Train Reca: {train_metrics.recall_list[-1]:.4f}\n"
            f"Val Loss: {val_metrics.loss_list[-1]:.4f}\n"
            f"Val Dice: {val_metrics.dice_list[-1]:.4f}\n"
            f"Val Accu: {val_metrics.accuracy_list[-1]:.4f}\n"
            f"Val Sens: {val_metrics.sensitivity_list[-1]:.4f}\n"
            f"Val Spec: {val_metrics.specificity_list[-1]:.4f}\n"
            f"Val Prec: {val_metrics.precision_list[-1]:.4f}\n"
            f"Val Reca: {val_metrics.recall_list[-1]:.4f}\n"
            f"Val3D Loss: {val_3d_metrics.loss_list[-1]:.4f}\n"
            f"Val3D Dice: {val_3d_metrics.dice_list[-1]:.4f}\n"
            f"Val3D Accu: {val_3d_metrics.accuracy_list[-1]:.4f}\n"
            f"Val3D Sens: {val_3d_metrics.sensitivity_list[-1]:.4f}\n"
            f"Val3D Spec: {val_3d_metrics.specificity_list[-1]:.4f}\n"
            f"Val3D Prec: {val_3d_metrics.precision_list[-1]:.4f}\n"
            f"Val3D Reca: {val_3d_metrics.recall_list[-1]:.4f}"
        )
        self.print_log(log_text)

    def print_log(self, log_text):
        self.logger.log(msg=log_text, level=logging.INFO)
        print(log_text)

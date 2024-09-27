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
        epoch = len(train_metrics.history["dice"])

        def format_metric_tuple(metric_tuple):
            # Convert tuple values to a string with 4 decimal points
            return ', '.join([f"{value:.4f}" for value in metric_tuple])

        def format_metrics(label, metrics):
            # Helper function to format a block of metrics
            return (
                f"{label} Loss: {metrics.history['loss'][-1][0]:.4f}\n"
                f"{label} Dice: {format_metric_tuple(metrics.history['dice'][-1])}\n"
                f"{label} Accu: {format_metric_tuple(metrics.history['accuracy'][-1])}\n"
                f"{label} Sens: {format_metric_tuple(metrics.history['sensitivity'][-1])}\n"
                f"{label} Spec: {format_metric_tuple(metrics.history['specificity'][-1])}\n"
                f"{label} Prec: {format_metric_tuple(metrics.history['precision'][-1])}\n"
                f"{label} Reca: {format_metric_tuple(metrics.history['recall'][-1])}"
            )

        log_text = (
            f"Epoch {epoch}/{epochs}\n" +
            format_metrics("Train", train_metrics) + "\n" +
            format_metrics("Val", val_metrics) + "\n" +
            format_metrics("Val3D", val_3d_metrics)
        )

        self.print_log(log_text)

    def print_log(self, log_text):
        self.logger.log(msg=log_text, level=logging.INFO)
        print(log_text)

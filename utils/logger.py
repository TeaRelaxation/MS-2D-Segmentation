import logging
import os
import torch


class Logger:
    def __init__(self, root_dir, experiment_name):
        self.path = os.path.join(root_dir, experiment_name)
        os.makedirs(self.path, exist_ok=True)
        log_path = os.path.join(self.path, "logs.log")
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')

    def save_model(self, model, filename):
        model_path = os.path.join(self.path, filename)
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved on {model_path}")

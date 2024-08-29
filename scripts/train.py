import argparse
import sys
import torch
from torch import optim

sys.path.insert(0, '.')

from dataloaders.select_data import select_data
from models.select_model import select_model
from utils.logger import Logger
from utils.trainer import Trainer

parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
parser.add_argument('--experiment_name', type=str, default="Experiment", help='Name of the experiment')
parser.add_argument('--model_name', type=str, default="UNet", help='Name of the model to use')
parser.add_argument('--dataset_name', type=str, default="MS", help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default="../datasets/MS", help='Path to the dataset')
parser.add_argument('--logs_path', type=str, default="../logs", help='Path to save logs')
parser.add_argument('--n_classes', type=int, default=5, help='Number of classes')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = select_model(args.model_name, n_classes=args.n_classes)
train_data, val_data = select_data(args.dataset_name, args.dataset_path)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
logger = Logger(root_dir=args.logs_path, experiment_name=args.experiment_name)

trainer = Trainer(
    train_dataset=train_data,
    val_dataset=val_data,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=args.epochs,
    batch_size=args.batch_size,
    device=device,
    logger=logger
)

trainer.train()

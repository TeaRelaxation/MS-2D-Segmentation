import argparse
import sys
import torch
from torch import optim

sys.path.insert(0, '.')

from dataloaders.select_data import select_data
from models.select_model import select_model
from utils.logger import Logger
from utils.trainer import Trainer
from losses.select_loss import select_loss

parser = argparse.ArgumentParser(description='Train a model with specified parameters.')
parser.add_argument('--experiment', type=str, default="Exp1", help='Name of the experiment')
parser.add_argument('--model', type=str, default="DeepLabV3Plus_ResNet34", help='Name of the model to use')
parser.add_argument('--dataset', type=str, default="MS", help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default="../datasets/MS", help='Path to the dataset')
parser.add_argument('--logs_path', type=str, default="../logs", help='Path to save logs')
parser.add_argument('--n_classes', type=int, default=5, help='Number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for the optimizer')
parser.add_argument('--loss', type=str, default="WCEDiceFocal", help='Loss function')
parser.add_argument('--height', type=int, default=224, help='Height of image')
parser.add_argument('--width', type=int, default=192, help='Width of image')
parser.add_argument('--resize_type', type=str, default="pad", help='Use pad or resize')
parser.add_argument('--workers', type=int, default=0, help='Number of CPU workers')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = select_model(args.model, n_classes=args.n_classes)
train_data, val_data = select_data(args.dataset, args.dataset_path, args.height, args.width, args.resize_type)
criterion = select_loss(args.loss, device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*1, eta_min=1e-5)
logger = Logger(root_dir=args.logs_path, experiment_name=args.experiment)

trainer = Trainer(
    train_dataset=train_data,
    val_dataset=val_data,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=args.epochs,
    batch_size=args.batch_size,
    device=device,
    logger=logger,
    n_classes=args.n_classes,
    workers=args.workers
)

trainer.train()

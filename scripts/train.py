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
parser.add_argument('--workers', type=int, default=0, help='Number of CPU workers')
parser.add_argument('--max_pixel', type=float, default=375.3621, help='Max pixel value for scaling to 0 to 1')
parser.add_argument('--mean', type=float, default=0.1266, help='Mean for normalization')
parser.add_argument('--std', type=float, default=0.1360, help='Std for normalization')
parser.add_argument('--crop_h', type=int, default=128, help='Height of image for training')
parser.add_argument('--crop_w', type=int, default=128, help='Width of image for training')
parser.add_argument('--infer_h', type=int, default=224, help='Height of image for inference')
parser.add_argument('--infer_w', type=int, default=192, help='Width of image for inference')
parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels (1 or 3)')
parser.add_argument('--is_imagenet', type=str, default="False", help='Use ImageNet mean and std (True or False)')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = select_loss(args.loss, device)
logger = Logger(root_dir=args.logs_path, experiment_name=args.experiment)

model = select_model(
    args.model,
    n_classes=args.n_classes,
    in_channels=args.in_channels,
    is_imagenet=args.is_imagenet
)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*1, eta_min=1e-5)

train_data, val_data = select_data(
    dataset_name=args.dataset,
    dataset_path=args.dataset_path,
    max_pixel=args.max_pixel,
    mean=args.mean,
    std=args.std,
    crop_h=args.crop_h,
    crop_w=args.crop_w,
    infer_h=args.infer_h,
    infer_w=args.infer_w,
    in_channels=args.in_channels,
    is_imagenet=args.is_imagenet
)

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

import torch
import torch.optim as optim
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from src.MSDNDataset import MSDNDataset
import argparse as ap 

def get_dataloaders(transform = None) -> dict[str, dict[str, DataLoader]]:
    data_folders = ['train', 'val', 'test']
    img_types = ['noisy', 'clean', 'label']

    datasets = {
        data_folder: {img_type: MSDNDataset(data_folder, img_type, transform) for img_type in img_types}
        for data_folder in data_folders
    }

    dataloaders = {
        data_folder: {img_type: DataLoader(datasets[data_folder][img_type], batch_size=4, shuffle=True) for img_type in img_types}
        for data_folder in data_folders
    }

    return dataloaders

def main():

    parser = ap.ArgumentParser(description="Entry point for the training of the MSDN dataset.")

    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--lr", "-l", type=float, default=0.001, help="Learning rate for the model.")
    
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr

    transform = None  # TODO: Add transforms
    dataloaders = get_dataloaders(transform)


    


if __name__ == "__main__":
    main()
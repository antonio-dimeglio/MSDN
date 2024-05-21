import argparse as ap 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.MSDNDataset import MSDNDataset
from src.Model import MSDN
import matplotlib.pyplot as plt

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
    
    model = MSDN(in_channels=3, num_classes=10)
    # Loss metric: RMSE (or SSIM?)
    criterion = nn.MSE() # calcualte the sqrt(MSE) in the training loop
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for sinograms, images in dataloaders['train']['clean']:
            # Model should map FBP reconsutruction of sinogram to image
            # TODO: obtain FBP reconstructions of sinograms -> do it on more outer level - no need to do it in the training loop

            # optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.sqrt(criterion(outputs, images)) # square root of MSE = RMSE

            # loss.backward()
            # optimizer.step()


    


if __name__ == "__main__":
    main()
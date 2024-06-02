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
from tqdm import trange
from src.MSDNNet import MSDNet

def get_dataloaders(root_folder, transform = None) -> dict[str, dict[str, DataLoader]]:
    splits = ['train', 'val', 'test']
    groups = ['noisy', 'clean']
    num_angles = [45, 90, 180, 256]


    datasets = {
        split: {angle: {group: MSDNDataset(f'{root_folder}/fbps/{split}/{angle}/{group}',
                                           f'{root_folder}/phantoms/{split}/{angle}/{group}',
                                           transform)
                        for group in groups} 
                for angle in num_angles}
        for split in splits
    }

    dataloaders = {
        split: {angle: {group: DataLoader(datasets[split][angle][group], batch_size=4, shuffle=True)
                        for group in groups}
                for angle in num_angles}
        for split in splits
    }

    return dataloaders

def main():

    parser = ap.ArgumentParser(description="Entry point for the training of the MSDN dataset.")

    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--lr", "-l", type=float, default=0.001, help="Learning rate for the model.")
    
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.lr

    root_folder = '3d_data'
    transform = None  # TODO: Add transforms
    dataloaders = get_dataloaders(root_folder, transform)
    
    model = MSDNet(in_channels=4,
                   out_channels=4,
                   num_features=4,
                   num_layers=10, 
                   dilations=np.arange(1, 11))

    # Loss metric: RMSE (or SSIM?)
    criterion = nn.MSELoss() # calcualte the sqrt(MSE) in the training loop
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloaders['train'][256]['clean']:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloaders['train'][256]['clean']):.4f}")



    


if __name__ == "__main__":
    main()
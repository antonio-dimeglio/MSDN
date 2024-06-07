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
from tqdm import tqdm
from src.MSDNNet import MSDNet


def get_dataloaders(root_folder, transform = None) -> dict[str, dict[str, DataLoader]]:
    splits = ['train', 'test']
    groups = ['noisy', 'clean']
    num_angles = [45, 90, 180]


    datasets = {
        split: {group: {angle: MSDNDataset(f'{root_folder}/recon/{split}/{group}/{angle}',
                                           f'{root_folder}/phantom/{split}/{group}/{angle}',
                                           transform)
                        for angle in num_angles} 
                for group in groups}
        for split in splits
    }

    dataloaders = {
        split: {group: {angle: DataLoader(datasets[split][group][angle], batch_size=4, shuffle=True)
                        for angle in num_angles}
                for group in groups}
        for split in splits
    }

    return dataloaders


def main():
    
    parser = ap.ArgumentParser(description="Entry point for the training of the MSDN dataset.")

    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs to train the model.")
    parser.add_argument("--lr", "-l", type=float, default=0.01, help="Learning rate for the model.")
    
    args = parser.parse_args()

    num_epochs = args.epochs
    lr = args.lr


    root_folder = 'data'
    transform = None 
    dataloaders = get_dataloaders(root_folder, transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_layers = 100
    
    model = MSDNet(in_channels=4,
                   out_channels=4,
                   num_features=4,
                   num_layers=num_layers,
                   dilations=1 + (np.arange(num_layers + 1) % 10),
                   ).to(device)

    # Loss metric: RMSE (or SSIM?)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training model...")
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for inputs, targets in dataloaders['train']['clean'][180]:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.type(torch.float32)
            loss = criterion(outputs, targets)
            loss = torch.sqrt(loss) # RMSE
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloaders['train']['clean'][180]):.4f}")

    print("Model trained.")
    print("Saving model...")
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

from src.MSDNDataset import MSDNDataset
from src.MSDNNet import MSDNet

def get_test_dataloaders(root_folder,
                         transform = None) -> dict[str, dict[str, DataLoader]]:
    groups = ['noisy', 'clean']
    num_angles = [45, 90, 180]


    datasets = {
        group: {angle: MSDNDataset(f'{root_folder}/recon/test/{group}/{angle}',
                                   f'{root_folder}/phantom/test/{group}/{angle}',
                                   transform)
                for angle in num_angles} 
        for group in groups
    }

    dataloaders = {
        group: {angle: DataLoader(datasets[group][angle],
                                  batch_size=4,
                                  shuffle=True)
                for angle in num_angles}
        for group in groups
    }

    return dataloaders


def main():
    print('Loading model.')
    model = MSDNet(in_channels=4, 
                   out_channels=4, 
                   num_features=4, 
                   num_layers=100,  
                   dilations=np.arange(1, 101))
    
    model.load_state_dict(torch.load('model.pth',
                                     map_location=torch.device('cpu')))
    # summary(model)
    print('Model loaded succesfully.\nLoading datasets...')

    dataloaders = get_test_dataloaders('.', None)
    print('Datasets loaded succesfully.')

    criterion = nn.MSELoss() 

    total_loss = 0
    counter = 0
    for inputs, targets in dataloaders['clean'][45]:
        if inputs.size()[0] % 4 != 0:
            break
        outputs = model(inputs)
        [iio.imsave(f'eval/output_{counter * 4 + i}.tiff', 
                    outputs[i].detach().numpy())
         for i in range(len(outputs))]
        [iio.imsave(f'eval/input_{counter * 4 + i}.tiff',
                    inputs[i].detach().numpy())
         for i in range(len(outputs))]
        targets = targets.type(torch.float32)
        loss = criterion(outputs, targets)
        loss = torch.sqrt(loss) # RMSE
        total_loss += loss.item()
        counter += 1
    
    print(f"Loss: {total_loss/len(dataloaders['clean'][45]):.4f}")




if __name__ == '__main__':
    main()
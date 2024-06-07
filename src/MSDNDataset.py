from torch.utils.data import Dataset 
import os 
import numpy as np
import imageio.v2 as imageio
import torch 


class MSDNDataset(Dataset):
    """
        Dataset class for the MSDN dataset.

        This dataset class can be used to load the generated FBP reconstructions
        and the corresponding phantom images.
        It allows for the loading of the data in the form of numpy arrays,
        starting from .tiff files.
    """

    def __init__(self, fbp_folder:str, phantom_folder:str, transform=None):
        """
            Constructor for the MSDNDataset class.

            Args:
                data_folder (str): Path to the folder containing FBP
                                reconstructions (i.e. the training data).
                phantom_folder (str): Path to the folder containing the origina
                                     phantom images (i.e. the ground truth).
                transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.transform = transform
        self.fbp_folder = fbp_folder 
        self.phantom_folder = phantom_folder
        self.fbp_files = sorted(os.listdir(self.fbp_folder))
        self.phantom_files = sorted(os.listdir(self.phantom_folder))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.fbp_files)
    
    def __getitem__(self, idx):
        fbp = imageio.imread(os.path.join(self.fbp_folder, self.fbp_files[idx]))
        phantom = imageio.imread(os.path.join(self.phantom_folder, self.phantom_files[idx]))

        if self.transform:
            fbp = self.transform(fbp)
            phantom = self.transform(phantom)

        fbp = torch.tensor(fbp, dtype=torch.float32).to(self.device)
        phantom = torch.tensor(phantom, dtype=torch.float32).to(self.device)
        
        return fbp, phantom
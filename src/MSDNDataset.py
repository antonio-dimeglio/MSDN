from torch.utils.data import Dataset 
import os 
import numpy as np
import imageio.v2 as imageio
import torch 


class MSDNDataset(Dataset):
    """
        Dataset class for the MSDN dataset.

        This dataset class can be used to load the generated sinograms and the corresponding
        phantom images.
        It allows for the loading of the data in the form of numpy arrays, starting frmo .tiff files.
    """

    def __init__(self, data_folder:str, img_type:str, transform=None):
        """
            Constructor for the MSDNDataset class.

            Args:
                data_folder (str): The folder containing the data, can be the train, val or test folder.
                img_type (str): The type of image to load, can be 'noisy', 'clean' or 'label'.
                transform (callable, optional): Optional transform to be applied on a sample.
        """
        if data_folder not in ['train', 'val', 'test']:
            raise ValueError('data_folder should be one of "train", "val" or "test"')
        if img_type not in ['noisy', 'clean', 'label']:
            raise ValueError('img_type should be one of "noisy", "clean" or "label"')

        self.transform = transform
        self.data_folder = data_folder 
        self.img_type = img_type

        self.sinogram_folder = os.path.join("sinogram", self.data_folder, self.img_type)
        self.image_folder = os.path.join("phantom", self.data_folder, self.img_type)

        self.sinogram_files = os.listdir(self.sinogram_folder)
        self.image_files = os.listdir(self.image_folder)

    def __len__(self):
        # Both the sinogram and image files should have the same length,
        # so it is enough to return the length of one of them.
        return len(self.sinogram_files)
    
    def __getitem__(self, idx):
        sinogram = imageio.imread(os.path.join(self.sinogram_folder, self.sinogram_files[idx]))
        image = imageio.imread(os.path.join(self.image_folder, self.image_files[idx]))

        if self.transform:
            sinogram = self.transform(sinogram)
            image = self.transform(image)
        
        return sinogram, image
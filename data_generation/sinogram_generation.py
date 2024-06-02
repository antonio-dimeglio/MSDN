import astra 
import numpy as np
import imageio.v2 as iio
from pathlib import Path
from time import time
import argparse
import os  
import matplotlib.pyplot as plt

def generate_sinograms():
    folders = ['train', 'test', 'val']
    angles = [180, 90, 45]
    image_type = ['noisy', 'clean', 'label']
    

    for folder in folders:

        # Create folder for sinograms
        sinogram_folder = Path(f'sinogram/{folder}')
        os.makedirs(sinogram_folder, exist_ok=True)
        for img_type in image_type:
            sinogram_type_folder = sinogram_folder / img_type
            os.makedirs(sinogram_type_folder, exist_ok=True)

            num_images = len(os.listdir(f'phantom/{folder}/{img_type}'))
            for angle in angles:
                os.makedirs(sinogram_type_folder / str(angle), exist_ok=True)
            

            for i in range(num_images):
                data = iio.imread(f'phantom/{folder}/{img_type}/{i}.tiff')

                for angle in angles:
                    
                    proj_geom = astra.create_proj_geom('parallel', 1.0, data.shape[0], np.linspace(0, angle, data.shape[0], False))
                    vol_geom = astra.create_vol_geom(data.shape)
                    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
                    id, sinogram = astra.create_sino(data, proj_id)

                    sinogram = sinogram / sinogram.max()

                    iio.imwrite(sinogram_type_folder / str(angle) / f'{i}.tiff', sinogram)
                    print(f'Sinogram {i} of {folder}/{img_type} generated.')

def __main__():
    generate_sinograms()

if __name__ == '__main__':
    __main__()
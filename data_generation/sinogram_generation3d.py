import astra 
import numpy as np
import imageio.v2 as iio
import os  
from tqdm import trange
from helper import convert_rgba_to_greyscale, convert_greyscale_to_rgba
import matplotlib.pyplot as plt

def generate_sinogram(phantom_filename, out_filename, num_angles):
    angles = np.linspace(0, num_angles, data.shape[0], False)
    data = iio.imread(phantom_filename)
    if len(np.array(data).shape) == 3:
        data = convert_rgba_to_greyscale(data)
    proj_geom = astra.create_proj_geom('parallel', 1.0, data.shape[0], angles)
    vol_geom = astra.create_vol_geom(data.shape)
    proj_id = astra.create_projector('linear', proj_geom, vol_geom)
    _, sinogram = astra.create_sino(data, proj_id)
    
    sinogram = (sinogram - sinogram.min()) / (sinogram.max() - sinogram.min())
    iio.imsave(out_filename, sinogram)

def __main__():
    root = '3d_data/slices/'
    out_root = '3d_data/sinograms/'
    os.makedirs(out_root, exist_ok=True)
    filenames = os.listdir(root)
    for i in trange(len(filenames)):
        for angle in (45, 90, 180):
            generate_sinogram(root + filenames[i],
                              out_root + f'sinogram_{i}_{angle}.tiff',
                              num_angles=angle)

if __name__ == '__main__':
    __main__()
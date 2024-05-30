import imageio.v2 as iio
import numpy as np
import os
from tqdm import trange

from phantom_3d import generate_phantom
from sinogram_reconstruction3d import create_sinogram, reconstruct
from helper import get_slices, load_phantom


def main():
    data_folder = '3d_data_2'
    num_slices = 128
    phantom_fn = '3d_data/phantom_256_256_128.npy'
    phantom_fn = None
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(f'{data_folder}/slices', exist_ok=True)
    os.makedirs(f'{data_folder}/sinograms', exist_ok=True)
    os.makedirs(f'{data_folder}/fbp_reconstructions', exist_ok=True)

    # Stage 1: Generate a phantom and slice it
    if phantom_fn:
        phantom = load_phantom(phantom_fn, (256, 256, 128))
    else:
        phantom_shape = np.array([1024, 1024, num_slices])
        phantom = generate_phantom(phantom_shape, 10000, 1000, 50,
                                   save=True, out_file=f'{data_folder}/phantom_1024_1024_{num_slices}.npy')
    
    slices = get_slices(phantom, rgba=False)

    # Stage 2: Create Sinograms, FBP reconstructions and save
    for i in trange(len(slices)):
        for angle in (45, 90, 180):
            sinogram = create_sinogram(slices[i], angle, num_iterations=100)
            fbp_recon = reconstruct(sinogram, angle, num_iterations=100)
            iio.imsave(f'{data_folder}/slices/phantom_3d_slice_{i}.tiff', slices[i])
            iio.imsave(f'{data_folder}/sinograms/phantom_3d_sinogram_{i}_{angle}.tiff', sinogram)
            iio.imsave(f'{data_folder}/fbp_reconstructions/phantom_3d_fbp_{i}_{angle}.tiff', fbp_recon)



if __name__ == '__main__':
    main()
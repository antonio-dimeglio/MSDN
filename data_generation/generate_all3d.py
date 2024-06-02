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
    # phantom_fn = None
    [os.makedirs(f'{data_folder}/{dt}/{split}/{angle}/{group}', exist_ok=True)
     for dt in ('phantoms', 'sinograms', 'fbps')
     for split in ('train', 'test', 'val')
     for angle in (45, 90, 180, 256)]


    # Stage 1: Generate a phantom and slice it
    if phantom_fn:
        phantom = load_phantom(phantom_fn, (256, 256, 128))
    else:
        phantom_shape = np.array([1024, 1024, num_slices])
        phantom = generate_phantom(phantom_shape, 10000, 1000, 50,
                                   save=True, out_file=f'{data_folder}/phantom_1024_1024_{num_slices}.npy')
    
    slices = get_slices(phantom, rgba=False)

    # Stage 2: Create Sinograms, FBP reconstructions and save
    for split in splits.keys():
        print(f'Starting {split}...')
        for angle in (45, 90, 180, 256):
            print(f'\t Angle {angle} in progress.')
            for i, slice in enumerate(splits[split]):
                sinogram = create_sinogram(slice, angle, num_iterations=100)
                fbp_recon = reconstruct(sinogram, angle, num_iterations=100)
                iio.imsave(f'{data_folder}/phantoms/{split}/{angle}/{group}/{i}.tiff', slice)
                iio.imsave(f'{data_folder}/sinograms/{split}/{angle}/{group}/{i}.tiff', sinogram)
                iio.imsave(f'{data_folder}/fbps/{split}/{angle}/{group}/{i}.tiff', fbp_recon)
            print(f'\t Angle {angle} finished.')
        print(f'Finished {split}.')





if __name__ == '__main__':
    main()
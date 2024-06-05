import imageio.v2 as iio
import numpy as np
import os
from tqdm import trange

from phantom_3d import generate_phantom
from sinogram_reconstruction3d import create_sinogram, reconstruct
from helper import get_slices, load_phantom


def main():
    data_folder = '3d_data'
    num_slices = 128
    phantom_fn = '3d_data/phantom_256_256_128.npy'
    # phantom_fn = None
    [os.makedirs(f'{data_folder}/{dt}/{split}/{angle}/{group}', exist_ok=True)
     for dt in ('phantoms', 'sinograms', 'fbps')
     for split in ('train', 'test', 'val')
     for angle in (45, 90, 180, 256)
     for group in ('noisy', 'clean')]


    # Stage 1: Generate a phantom and slice it
    if phantom_fn:
        phantom = load_phantom(phantom_fn, (256, 256, 128))
    else:
        phantom_shape = np.array([256, 256, num_slices])
        phantom = generate_phantom(phantom_shape, 1000, 1000, 50,
                                   save=True, out_file=f'{data_folder}/phantom_256_256_{num_slices}.npy')
    
    slices = get_slices(phantom, rgba=False)
    shuffled_idx = np.arange(len(slices))
    np.random.shuffle(shuffled_idx)
    # 60:30:10 split
    pivots = np.array([0.6 * len(shuffled_idx), 0.9 * len(shuffled_idx)]).astype(int)
    split_idx = {
        'train': shuffled_idx[:pivots[0]],
        'test': shuffled_idx[pivots[0]:pivots[1]],
        'val': shuffled_idx[pivots[1]:]
    }
    splits = {k: slices[split_idx[k]] for k in split_idx.keys()}

    # Stage 2: Create Sinograms, FBP reconstructions and save
    for split in splits.keys():
        print(f'Starting {split}...')
        for angle in (45, 90, 180, 256):
            print(f'\t Angle {angle} in progress.')
            for i, slice in enumerate(splits[split]):
                slice_noisy = slice + np.random.normal(size=slice.shape)
                sinogram = create_sinogram(slice, angle)
                fbp_recon = reconstruct(sinogram, angle, num_iterations=100)
                sinogram_noisy = create_sinogram(slice_noisy, angle)
                fbp_recon_noisy = reconstruct(sinogram_noisy, angle, num_iterations=100)
                iio.imsave(f'{data_folder}/phantoms/{split}/{angle}/clean/{i}.tiff', slice)
                iio.imsave(f'{data_folder}/sinograms/{split}/{angle}/clean/{i}.tiff', sinogram)
                iio.imsave(f'{data_folder}/fbps/{split}/{angle}/clean/{i}.tiff', fbp_recon)
                iio.imsave(f'{data_folder}/phantoms/{split}/{angle}/noisy/{i}.tiff', slice_noisy)
                iio.imsave(f'{data_folder}/sinograms/{split}/{angle}/noisy/{i}.tiff', sinogram_noisy)
                iio.imsave(f'{data_folder}/fbps/{split}/{angle}/noisy/{i}.tiff', fbp_recon_noisy)
            print(f'\t Angle {angle} finished.')
        print(f'Finished {split}.')





if __name__ == '__main__':
    main()
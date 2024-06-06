import argparse
import imageio.v2 as iio
import numpy as np
import os
from tqdm import trange
import re

from phantom_3d import generate_phantom
from sinogram_reconstruction3d import create_sinogram, reconstruct
from helper import get_slices, load_phantom


def parse_phantom_filename(filename):
    match = re.search(r'(\d+)_(\d+)_(\d+)', filename)
    if len(match.groups()) != 3:
        print('Was not able to parse filename.')
        return None
    shapes = [int(i) for i in match.groups()]
    return np.array(shapes)


def get_phantom(data_folder, phantom_filepath, phantom_shape):
    phantom = None
    if phantom_shape is None and phantom_filepath is None:
        print('Either filepath to phantom or desired phantom shape needed.')
        return None
    
    if phantom_shape is None:
        phantom_shape = parse_phantom_filename(phantom_filepath)

    if phantom_filepath is None:
        out_file  = (f'{data_folder}/phantom_{phantom_shape[0]}'
                     f'_{phantom_shape[1]}_{phantom_shape[2]}.npy')
        phantom = generate_phantom(phantom_shape, 1000, 1000, 50, save=True, 
                                   out_file=out_file)
    else: 
        phantom = load_phantom(phantom_filepath, phantom_shape)

    return phantom


def generate_data(data_folder='3d_data', phantom_filepath = None,
                  phantom_shape=None, group='clean'):
    # Stage 0: Initialize file structure
    [os.makedirs(f'{data_folder}/{dt}/{split}/{angle}/{group}', exist_ok=True)
     for dt in ('phantoms', 'sinograms', 'fbps')
     for split in ('train', 'test', 'val')
     for angle in (45, 90, 180, 256)]
    
    # Stage 1: Obtain phantom and slice it
    phantom = get_phantom(data_folder, phantom_filepath, phantom_shape)
    slices = get_slices(phantom, rgba=False)

    # Stage 2: Split intro train/test/val
    shuffled_idx = np.arange(len(slices))
    np.random.shuffle(shuffled_idx)
    # 60:30:10 split
    pivots = np.array(
        [0.6 * len(shuffled_idx), 0.9 * len(shuffled_idx)]
        ).astype(int)
    split_idx = {'train': shuffled_idx[:pivots[0]], 
                 'test': shuffled_idx[pivots[0]:pivots[1]],
                 'val': shuffled_idx[pivots[1]:]}
    splits = {k: slices[split_idx[k]] for k in split_idx.keys()}

    # Stage 3: Create Sinograms, FBP reconstructions and save
    for split in splits.keys():
        print(f'Starting {split}...')
        for angle in (45, 90, 180, 256):
            print(f'\t Angle {angle} in progress.')
            for i, slice in enumerate(splits[split]):
                if group == 'noisy':
                    slice = slice + np.random.normal(size=slice.shape)
                sinogram = create_sinogram(slice, angle)
                fbp_recon = reconstruct(sinogram, angle, num_iterations=100)
                iio.imsave(
                    f'{data_folder}/phantoms/{split}/{angle}/{group}/{i}.tiff',
                    slice)
                iio.imsave(
                    f'{data_folder}/sinograms/{split}/{angle}/{group}/{i}.tiff',
                    sinogram)
                iio.imsave(
                    f'{data_folder}/fbps/{split}/{angle}/{group}/{i}.tiff',
                    fbp_recon)
            print(f'\t Angle {angle} finished.')
        print(f'Finished {split}.')


def main():
    description=('This program prepares data for training the Mixed-Scale '
                 'Dense Network. It takes a 3d phantom (or optionally '
                 'creates it), creates its sinograms and FBP reconstructions. '
                 'Then, the files are divided depending on reconstruction '
                 'angle, train/test/val split, data type (slice, sinogram, '
                 'FBP reconstruction) and level of noise (clean/noisy).')
    parser = argparse.ArgumentParser(prog='Reconstructor',
                                     description=description)
    parser.add_argument('--data_folder', '-df', required=False,
                        default='3d_data',
                        help=('The path to the folder where '
                              'the data should be stored.'))
    parser.add_argument('--phantom_filename', '-pf', required=False,
                        help='The path to the pre-generated phantom.')
    parser.add_argument('--phantom_shape', '-ps', type=int, nargs=3,
                        help='an integer for the accumulator')
    parser.add_argument('--noisy', '-n', action='store_const', const='noisy',
                        default='clean',
                        help='an integer for the accumulator')
    args = parser.parse_args()
    phantom_shape = np.array(args.phantom_shape)
    generate_data(args.data_folder, args.phantom_filename, phantom_shape,
                  group=args.noisy)
    

if __name__ == '__main__':
    main()
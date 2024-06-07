import argparse
from datetime import datetime
import imageio.v2 as iio
import numpy as np
import os
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


def generate_data(data_folder='3d_data', phantom_shape=None,
                  generate_phantoms=False, group='clean'):
    # Stage 0: Initialize file structure
    [os.makedirs(f'{data_folder}/{dt}/{split}/{angle}/{group}', exist_ok=True)
     for dt in ('phantoms', 'sinograms', 'fbps')
     for split in ('train', 'test', 'val')
     for angle in (45, 90, 180)]
    
    # Stage 1: Obtain phantom and slice it
    phantoms = []
    phantom = None
    if generate_phantoms:
        for i in range(5):
            datetime_pattern = datetime.now().strftime('%y%m%d%H%M%S')
            out_file  = (f'{data_folder}/source_phantoms/phantom_'
                         f'{phantom_shape[0]}_{phantom_shape[1]}_'
                         f'{phantom_shape[2]}-{datetime_pattern}.npy')
            p = generate_phantom(phantom_shape, 1000, 1000, 50,
                                 save=True, out_file=out_file)
            phantoms.append(p)
    
    else:
        phantom_folder = f'{data_folder}/source_phantoms'
        for fn in os.listdir(phantom_folder):
            full_fn = f'{phantom_folder}/{fn}'
            phantom_shape = parse_phantom_filename(fn)
            p = load_phantom(full_fn, phantom_shape)
            phantoms.append(p)
        
    phantom = np.concatenate(phantoms, axis=2)
    print(f'Obtained 3D phantom of size {phantom.shape}.')
    slices = get_slices(phantom, rgba=False)

    # Stage 2: Split intro train/test/val
    shuffled_idx = np.arange(len(slices))
    np.random.shuffle(shuffled_idx)
    # 70:30:0 split
    pivots = np.array([0.7 * len(shuffled_idx), len(shuffled_idx)]).astype(int)
    split_idx = {'train': shuffled_idx[:pivots[0]], 
                 'test': shuffled_idx[pivots[0]:pivots[1]],
                 'val': shuffled_idx[pivots[1]:]}
    splits = {k: slices[split_idx[k]] for k in split_idx.keys()}

    # Stage 3: Create Sinograms, FBP reconstructions and save
    for split in splits.keys():
        print(f'Starting {split}...')
        for angle in (45, 90, 180):
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
    parser.add_argument('--generate_phantoms', '-gp', action='store_true',
                        help=('When called, the program '
                              'generates source phantoms first.'))
    parser.add_argument('--noisy', '-n', action='store_const', const='noisy',
                        default='clean',
                        help='When called, noise is applied to slices.')
    args = parser.parse_args()
    phantom_shape = np.array([256, 256, 128])
    generate_data(args.data_folder, phantom_shape,
                  args.generate_phantoms, group=args.noisy)
    

if __name__ == '__main__':
    main()
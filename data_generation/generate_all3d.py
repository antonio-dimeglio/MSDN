import numpy as np
from tqdm import trange

from phantom_3d import generate_phantom
from sinogram_reconstruction3d import reconstruct_with_angles
from helper import save_slices


def main():
    data_folder = '3d_data'
    num_slices = 128
    # Stage 1: Generate a phantom and slice it
    phantom_shape = np.array([256, 256, num_slices])
    phantom = generate_phantom(phantom_shape, 1000, 1000, 50, True)
    save_slices(phantom, data_folder)

    # Stage 2: Create sinograms and FBP reconstructions
    for i in trange(num_slices):
        reconstruct_with_angles(f'3d_data/slices/phantom_3d_slice_{i}.tiff', data_folder)

if __name__ == '__main__':
    main()
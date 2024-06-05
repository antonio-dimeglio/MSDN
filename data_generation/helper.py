import imageio
import numpy as np
import os
from tqdm import trange

def load_phantom(filename, phantom_size=None):
    phantom = np.load(filename)
    phantom = np.unpackbits(phantom)
    return phantom.reshape(phantom_size)

def save_slices(phantom3d, out_folder):
    os.makedirs(f'{out_folder}/slices', exist_ok=True)
    for i in trange(phantom3d.shape[-1]):
        slice = phantom3d[:, :, i]
        slice = slice.reshape(slice.shape[0], slice.shape[1], 1)
        slice = np.repeat(slice, 4, axis=2) # Convert to 4 channels per px
        slice[:, :, 3] = 255 # Alpha channel always max
        slice[:, :, :3] *= 255 # RGB channels rescaled to 0..255
        imageio.imsave(f'{out_folder}/slices/phantom_3d_slice_{i}.tiff', slice)

def get_slices(phantom3d, rgba=False):
    slices = []
    for i in trange(phantom3d.shape[-1]):
        slice = phantom3d[:, :, i]
        if rgba:
            slice = slice.reshape(slice.shape[0], slice.shape[1], 1)
            slice = np.repeat(slice, 4, axis=2) # Convert to 4 channels per px
            slice[:, :, 3] = 255 # Alpha channel always max
            slice[:, :, :3] *= 255 # RGB channels rescaled to 0..255
        slices.append(slice)
    return np.array(slices)
    

def convert_rgba_to_greyscale(img):
    rgb = img[:, :, :3]
    alpha = img[:, :, 3]
    return rgb.mean(axis=2) * alpha

def convert_greyscale_to_rgba(img):
    img = (img - img.min()) / (img.max() - img.min()) # Rescale to 0..1
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.repeat(img, 4, axis=2) # Convert to 4 channels per px
    img *= 255 # Rescale all channels to 0..255
    return img.astype(int)

def convert_binary_to_rgba(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.repeat(img, 4, axis=2) # Convert to 4 channels per px
    img[:, :, 3] = 255 # Alpha channel always max
    img[:, :, :3] *= 255 # RGB channels rescaled to 0..255
    return img.astype(int)
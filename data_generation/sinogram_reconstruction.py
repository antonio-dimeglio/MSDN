import astra
import imageio.v2 as iio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

def reconstruct_single(data, angle, i, recon_type_folder):
    angles = np.linspace(0, angle, data.shape[0], False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, data.shape[0], angles)

    vol_geom = astra.create_vol_geom(data.shape)
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, data=data)
    recon_id = astra.data2d.create('-vol', vol_geom, data=data)

    cfg = astra.astra_dict('FBP')
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    recon = astra.data2d.get(recon_id)
    iio.imwrite(recon_type_folder / str(angle) / f'{i}.tiff', recon)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)



def reconstruct_sinograms():
    folders = ['train', 'test']
    angles = [180, 90, 45]
    image_type = ['noisy', 'clean']


    for folder in folders:

        recon_folder = Path(f'recon/{folder}')
        os.makedirs(recon_folder, exist_ok=True)
        for img_type in image_type:
            recon_type_folder = recon_folder / img_type
            os.makedirs(recon_type_folder, exist_ok=True)


            for angle in angles:
                
                num_images = len(os.listdir(f'phantom/{folder}/{img_type}/{str(angle)}'))
                os.makedirs(recon_type_folder / str(angle), exist_ok=True)

                for i in range(num_images):
                    data = iio.imread(f'sinogram/{folder}/{img_type}/{angle}/{i}.tiff')

                    reconstruct_single(data, angle, i, recon_type_folder)
                    print(f'Reconstruction {i} of {folder}/{img_type}/{angle} done.')

def __main__():
    reconstruct_sinograms()

if __name__ == '__main__':
    __main__()
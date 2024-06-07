import astra
from helper import convert_rgba_to_greyscale
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import trange


def run_algorithm(image_id, sinogram_id,
                    projector_id, algorithm='FBP',
                    num_iterations=100):
    
    # Set up
    cfg = astra.astra_dict(algorithm)
    cfg['ProjectorId'] = projector_id
    cfg['ProjectionDataId'] = sinogram_id
    image_id_str = 'VolumeDataId' if algorithm == 'FP' else 'ReconstructionDataId'
    cfg[image_id_str] = image_id
    if algorithm == 'SIRT':
        cfg['option'] = {'MinConstraint': 0, 'MaxConstraint': 1}
    # Run the algorithm
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iterations)
    # Clean up
    astra.algorithm.delete(alg_id)

    return astra.data2d.get(image_id) 


def create_sinogram(image, num_angles, num_iterations=100):
    angles = np.linspace(0, num_angles, image.shape[0], False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[0], angles)
    vol_geom = astra.create_vol_geom(image.shape)
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)
    recon_id = astra.data2d.create('-vol', vol_geom, data=image)

    sinogram_id = astra.data2d.create('-sino', proj_geom)
    _ = run_algorithm(recon_id, sinogram_id, projector_id,
                        'FP', num_iterations)
    sinogram = astra.data2d.get(sinogram_id)

    # Clean-up
    astra.projector.delete(projector_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)

    return sinogram


def reconstruct(sinogram, num_angles, num_iterations=100):
    angles = np.linspace(0, num_angles, sinogram.shape[0], False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, sinogram.shape[1], angles)
    
    vol_geom = astra.create_vol_geom(sinogram.shape)
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)
    recon_id = astra.data2d.create('-vol', vol_geom, data=sinogram)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    # FBP
    fbp_recon = run_algorithm(recon_id, sinogram_id, projector_id,
                              'FBP', num_iterations)


    # Clean-up
    astra.projector.delete(projector_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)
    
    return fbp_recon


def reconstruct_with_angles(filename, output_dir, angles=(45, 90, 180),
                            num_iterations=100):
    image = iio.imread(filename)
    os.makedirs(f'{output_dir}/sinograms', exist_ok=True)
    os.makedirs(f'{output_dir}/fbp_reconstructions', exist_ok=True)
    fn = filename.replace('.tiff', '').split('/')[-1]
    if image.ndim == 3:
        image = convert_rgba_to_greyscale(image)

    sinograms, recons = {}, {'fbp': {}}
    for angle in angles:
        sino, fbp = reconstruct(image, angle, from_sinogram=False,
                                num_iterations=num_iterations)
        sinograms[angle] = sino
        recons['fbp'][angle] = fbp
        
    for angle in angles:
        iio.imsave(f'{output_dir}/sinograms/{fn}_sinogram_{angle}.tiff', sinograms[angle])
        iio.imsave(f'{output_dir}/fbp_reconstructions/{fn}_fbp{angle}.tiff', recons['fbp'][angle])

    return sinograms, recons


def main():

    for i in trange(128):
        reconstruct_with_angles(f'3d_data/slices/phantom_3d_slice_{i}.tiff', '3d_data')


if __name__ == '__main__':
    main()


    

        

    

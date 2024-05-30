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


def reconstruct(image, num_angles,
                from_sinogram=False,
                num_iterations=100):
    angles = np.linspace(0, np.pi, num_angles, False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[0], angles)
    
    vol_geom = astra.create_vol_geom(image.shape)
    projector_id = astra.create_projector('linear', proj_geom, vol_geom)
    recon_id = astra.data2d.create('-vol', vol_geom, data=image)

    if from_sinogram: # Assume image is sinogram
        sinogram = image.copy()
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    else: # Include sinogram creation
        sinogram_id = astra.data2d.create('-sino', proj_geom)
        _ = run_algorithm(recon_id, sinogram_id, projector_id,
                          'FP', num_iterations)
        sinogram = astra.data2d.get(sinogram_id)

    # FBP
    fbp_recon = run_algorithm(recon_id, sinogram_id, projector_id,
                              'FBP', num_iterations)


    # Clean-up
    astra.projector.delete(projector_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)
    
    return sinogram, fbp_recon


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

    # parser = argparse.ArgumentParser(
    #                 prog='Reconstructor',
    #                 description=('This program creates sinograms of an image with 45, '
    #                              '90, 180 angles then reconstructs the image using FBP.'
    #                              'Then stores the results in output folder'))
    # parser.add_argument('--filename', '-f', required=True,
    #                     help='The name of the file to be processed')
    # args = parser.parse_args()

    for i in trange(128):
        reconstruct_with_angles(f'3d_data/slices/phantom_3d_slice_{i}.tiff', '3d_data')


if __name__ == '__main__':
    main()


    

        

    


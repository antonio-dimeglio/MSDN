import astra
import imageio.v2
import numpy as np
from pathlib import Path
from time import time
import argparse
import matplotlib.pyplot as plt


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
    print(f"Starting FBP {num_angles}...")
    start = time()
    fbp_recon = run_algorithm(recon_id, sinogram_id, projector_id,
                              'FBP', num_iterations)
    print(f"FBP {num_angles} done {time() - start:.2f} seconds.")

    # SIRT
    print(f"Starting SIRT {num_angles}...")
    start = time()
    sirt_recon = run_algorithm(recon_id, sinogram_id, projector_id, 
                               'SIRT', num_iterations)
    print(f"SIRT {num_angles} done in {time() - start:.2f} seconds.")

    # Clean-up
    astra.projector.delete(projector_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(sinogram_id)
    
    return sinogram, fbp_recon, sirt_recon


def reconstruct_with_angles(filename, angles=(45, 90, 180),
                            num_iterations=100, save=True):
    image = imageio.v2.imread(filename)
    if image.ndim == 3:
        image = image.mean(axis=2).round().astype(int)
    main_folder = Path("outputs")
    direct_folder = filename.split('.')[0]
    direct_folder = direct_folder.split('/')[-1]
    (main_folder / direct_folder).mkdir(parents=True, exist_ok=True)
    root = f'{main_folder}/{direct_folder}'
    sinograms, recons = {}, {'fbp': {}, 'sirt': {}}
    for angle in angles:
        sino, fbp, sirt = reconstruct(image, angle, num_iterations=100)
        sinograms[angle] = sino
        recons['fbp'][angle] = fbp
        recons['sirt'][angle] = sirt

    if save:
        for angle in angles:
            plt.imsave(f'{root}/{direct_folder}_sino{angle}.tiff', sinograms[angle], cmap='gray')
            plt.imsave(f'{root}/{direct_folder}_fbp{angle}.tiff', recons['fbp'][angle], cmap='gray')
            plt.imsave(f'{root}/{direct_folder}_sirt{angle}.tiff', recons['sirt'][angle], cmap='gray')

    return sinograms, recons


def main():

    parser = argparse.ArgumentParser(
                    prog='Reconstructor',
                    description=('This program creates sinograms of an image with 45, '
                                 '90, 180 angles then reconstructs the image using FBP '
                                 'and SIRT. Then stores the results in output folder'))
    parser.add_argument('--filename', '-f', required=True,
                        help='The name of the file to be processed')
    args = parser.parse_args()
    
    reconstruct_with_angles(args.filename)


if __name__ == '__main__':
    main()


    

        

    


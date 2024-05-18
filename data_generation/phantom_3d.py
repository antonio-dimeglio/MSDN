import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

def generate_void(phantom_shape, voids, n_candidates=10_000, r_max=100):
    mid_point = phantom_shape // 2
    candidates = np.random.random((n_candidates, 3))
    candidates *= phantom_shape
    candidates = np.floor(candidates)

    max_radia = np.zeros(n_candidates)
    max_radia += r_max

    # Filter out points that aren't in the cyllinder
    c_distance = np.sum((candidates[:, :2] - mid_point[:2])**2, axis=1)**0.5
    # Assuming the axis of the cyllinder is in the '1st', 'y' plane
    candidates = candidates[c_distance < phantom_shape[0] / 2]
    max_radia = max_radia[c_distance < phantom_shape[0] / 2]
    c_distance = c_distance[c_distance < phantom_shape[0] / 2]

    # Find distance from the walls of the phantom
    h_distance = np.min([candidates[:, 2], phantom_shape[2] - candidates[:, 2],
                         candidates[:, 1], phantom_shape[1] - candidates[:, 1],
                         candidates[:, 0], phantom_shape[0] - candidates[:, 0]], axis=0)
    max_radia = np.min([h_distance, max_radia], axis=0)
    max_radia = np.min([mid_point[0] - c_distance, max_radia], axis=0)

    # Filter out points that overlap other voids
    if len(voids) > 0:
        min_p_distance = np.array([np.min(np.sqrt(np.sum((voids[:, :3] - cand[:3]) ** 2, axis=1)) - voids[:, 3])
                                   for cand in candidates])
        candidates = candidates[min_p_distance > 0]
        # No viable candidates found
        if len(candidates) < 1:
            return None
        max_radia = max_radia[min_p_distance > 0]
        min_p_distance = min_p_distance[min_p_distance > 0]
        max_radia = np.min([max_radia, min_p_distance], axis=0)

    baby_void_id = np.argmax(max_radia)
    baby_void = np.hstack([candidates[baby_void_id], max_radia[baby_void_id]])

    return baby_void.astype(int)


def generate_voids(phantom_shape, max_voids=10_000, n_candidates=1000, r_max=100):
    voids = generate_void(phantom_shape, np.array([]), n_candidates=n_candidates, r_max=r_max).reshape(1, 4)
    for _ in trange(max_voids - 1):
        bv = generate_void(phantom_shape, voids.copy(), n_candidates=n_candidates, r_max=r_max)
        if bv is None:
            continue
        voids = np.vstack([voids, bv])
    return voids


def create_sphere(matrix, void):
    x_0, y_0, z_0, r = void
    x_0 -= r
    y_0 -= r
    z_0 -= r
    r_2 = r ** 2
    center = (x_0 + r, y_0 + r, z_0 + r)
    for z in range(z_0, z_0 + 2*r):
        for y in range(center[1] - r, center[1] + 1):
            for x in range(center[0], center[0] + r + 1):
                d = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
                if  d >= r_2:
                    matrix[2*center[0] - x: x, y, z] = 1
                    matrix[2*center[0] - x: x, 2*center[1] - y - 1, z] = 1
                    break
    return matrix


def phantom_from_voids(phantom_shape, voids):
    phantom = np.zeros(phantom_shape)
    for i in trange(len(voids)):
        phantom = create_sphere(phantom, voids[i])
    return phantom


def generate_phantom(phantom_shape, max_voids=10_000, n_candidates=1000, r_max=100, save=False, out_file=None):
    print('Generating voids...')
    voids = generate_voids(phantom_shape, max_voids, n_candidates, r_max)
    print(f'{len(voids)} voids generated.')
    print('Rendering phantom...')
    phantom = phantom_from_voids(phantom_shape, voids)
    phantom = phantom.astype('uint8')
    print('Phantom rendered.')

    if save:
        if not out_file:
            out_file = f'phantom_{phantom_shape[0]}_{phantom_shape[1]}_{phantom_shape[2]}.npy'
        phantom_packed = np.packbits(phantom)
        np.save(out_file, phantom_packed)
        print(f'Phantom saved to {out_file}.')
    
    return phantom


def load_phantom(filename, phantom_size=None):
    phantom = np.load(filename)
    phantom = np.unpackbits(phantom)

    if not phantom_size:
        phantom_size = filename.replace('.npy', '').split('_')[1:]
        phantom_size = [int(i) for i in phantom_size]
    
    return phantom.reshape(phantom_size)


def main():
    phantom_shape = np.array([256, 256, 128])
    phantom = generate_phantom(phantom_shape, 1000, 1000, 50, True)
    plt.imshow(phantom[phantom_shape[0] // 2, :, :], origin='lower', cmap='grey')
    plt.savefig('phantom_X_axis.png')
    plt.clf()

    plt.imshow(phantom[:, phantom_shape[1] // 2, :], origin='lower', cmap='grey')
    plt.savefig('phantom_Y_axis.png')
    plt.clf()

    plt.imshow(phantom[:, :, phantom_shape[2] // 2], origin='lower', cmap='grey')
    plt.savefig('phantom_Z_axis.png')

    print('Example slice images saved to phantom_X_axis.png, phantom_Y_axis.png and phantom_Z_axis.png.')


if __name__ == '__main__':
    main()


    


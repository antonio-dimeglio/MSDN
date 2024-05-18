import numpy as np

def load_phantom(filename, phantom_size=None):
    phantom = np.load(filename)
    phantom = np.unpackbits(phantom)

    if not phantom_size:
        phantom_size = filename.replace('.npy', '').split('_')[1:]
        phantom_size = [int(i) for i in phantom_size]
    
    return phantom.reshape(phantom_size)
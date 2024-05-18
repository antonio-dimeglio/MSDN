import numpy as np 
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import argparse as ap

# CONSTANTS

NUM_TEMPLATES = 6
NUM_PIXELS = 256
NUM_TEMPLATES_IN_PHANTOM = 24 
TEMPLATE_SIZE = NUM_PIXELS // 8 

def generate_phantoms():
    image = np.zeros((NUM_PIXELS, NUM_PIXELS), dtype=np.float32)
    label = np.zeros((NUM_PIXELS, NUM_PIXELS), dtype=np.uint8)

    # Template definition
    template = np.zeros((NUM_TEMPLATES, TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=np.float32)
    # Full square
    template[0] = 1 
    
    # Hollow Square
    template[1] = 1 
    template[1][TEMPLATE_SIZE//4:-TEMPLATE_SIZE//4,TEMPLATE_SIZE//4:-TEMPLATE_SIZE//4]=0 

    # Filled circle
    xx,yy = np.mgrid[-1:1:1j*TEMPLATE_SIZE,-1:1:1j*TEMPLATE_SIZE]
    template[2] = xx**2+yy**2<1

    # Hollow circle
    template[3] = xx**2+yy**2<1
    template[3][xx**2+yy**2<0.25]=0

    # Filled triangle
    template[4] = np.triu(np.ones((TEMPLATE_SIZE, TEMPLATE_SIZE)))

    # Hollow triangle
    template[5] = np.triu(np.ones((TEMPLATE_SIZE, TEMPLATE_SIZE)))
    # TODO - Fill in the hollow triangle template

    # Image generation
    i = 0 # Counter
    tp = 0 # Type of pattern

    while i < NUM_TEMPLATES_IN_PHANTOM: 
        found = False
        while not found:
            x, y = (np.random.random(2) * (NUM_PIXELS - TEMPLATE_SIZE)).astype(np.int32)
            if label[x:x+TEMPLATE_SIZE, y:y+TEMPLATE_SIZE].max() == 0:
                found = True
            
        volume = np.random.random() * 0.8 + 0.2

        image[x:x+TEMPLATE_SIZE, y:y+TEMPLATE_SIZE] = template[tp] * volume
        label[x:x+TEMPLATE_SIZE, y:y+TEMPLATE_SIZE] = template[2 * (tp // 2)] * (tp + 1)

        tp = (tp + 1) % NUM_TEMPLATES

        i += 1

    noisy_image = image + np.random.normal(size=image.shape)

    return noisy_image, image, label


def main():
    parser = ap.ArgumentParser(description="Generate phantom images.")

    # Output folder
    parser.add_argument("--output", "-o", type=str, default="phantom", help="Output directory")
    
    # Number of training images
    parser.add_argument("--num_train", "-n", type=int, default=100, help="Number of training images")

    # Number of validation images
    parser.add_argument("--num_val", "-v", type=int, default=25, help="Number of validation images")

    # Number of test images
    parser.add_argument("--num_test", "-t", type=int, default=10, help="Number of test images")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    for folder in ["train", "val", "test"]:
        path = output_dir / folder
        path.mkdir(exist_ok=True)

        # Create subfolders for noisy, clean and label images
        for subfolder in ["noisy", "clean", "label"]:
            (path / subfolder).mkdir(exist_ok=True)

        # Generate images
        num_images = getattr(args, f"num_{folder}")

        for i in range(num_images):
            noisy_image, clean_image, label = generate_phantoms()

            imageio.imsave(path / "noisy" / f"{i}.tiff", noisy_image)
            imageio.imsave(path / "clean" / f"{i}.tiff", clean_image)
            imageio.imsave(path / "label" / f"{i}.tiff", label)
            print(f"Generated {folder} image {i + 1}/{num_images}")

    print("Done!")



if __name__ == "__main__":
    main()
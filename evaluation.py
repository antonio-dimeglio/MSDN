from datetime import datetime
import imageio.v2 as iio
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm

from src.MSDNDataset import MSDNDataset
from src.MSDNNet import MSDNet

def get_test_dataloaders(root_folder,
                         transform = None) -> dict[str, dict[str, DataLoader]]:
    groups = ['noisy', 'clean']
    num_angles = [45, 90, 180]


    datasets = {
        group: {angle: MSDNDataset(f'{root_folder}/recon/test/{angle}/{group}',
                                   f'{root_folder}/phantom/test/{angle}/{group}',
                                   transform)
                for angle in num_angles} 
        for group in groups
    }

    dataloaders = {
        group: {angle: DataLoader(datasets[group][angle], batch_size=4)
                for angle in num_angles} for group in groups}

    return dataloaders


def main():
    root_folder = '3d_data'
    model_fn = 'model3d.pth'
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    eval_dir = f"evaluations/{model_fn.replace('.pth', '')}_{timestamp}"
    os.makedirs(eval_dir)
    print('Loading model.')
    model = MSDNet(in_channels=4, out_channels=4, num_features=4, 
                   num_layers=100,  dilations=np.arange(1, 101))
    model.load_state_dict(torch.load(model_fn,
                                     map_location=torch.device('cpu')))
    print('Model loaded succesfully.\nLoading datasets...')

    dataloaders = get_test_dataloaders(root_folder, None)
    print('Datasets loaded succesfully.')

    metrics = {
        'ssim': {'msdn': 0, 'fbp': 0},
        'rmse': {'msdn': 0, 'fbp': 0}
    }
    counter = 0
    for inputs, targets in tqdm(dataloaders['clean'][45]):
        if inputs.size()[0] % 4 != 0:
            break
        outputs = model(inputs)
        targets = targets.type(torch.float32)

        for i in range(len(outputs)):
            ground_truth = iio.imread(
                f'{root_folder}/phantom/test/45/clean/{counter * 4 + i}.tiff')
            output = outputs[i].detach().numpy()
            input = inputs[i].detach().numpy()
            ssim_msdn = ssim(ground_truth, output,
                             data_range=output.max() - output.min())
            if np.isnan(ssim_msdn):
                ssim_msdn = metrics['ssim']['msdn'] / (counter * 4 + i)
            metrics['ssim']['msdn'] += ssim_msdn
            metrics['rmse']['msdn'] += np.sqrt(mse(output, ground_truth))
            ssim_fbp = ssim(ground_truth, input,
                            data_range=input.max() - input.min())
            if np.isnan(ssim_fbp):
                ssim_fbp = metrics['ssim']['fbp'] / (counter * 4 + i)
            metrics['ssim']['fbp'] += ssim_fbp
            metrics['rmse']['fbp'] += np.sqrt(mse(input, ground_truth))
            
            iio.imsave(f'{eval_dir}/output_{counter * 4 + i}.tiff', output)
            iio.imsave(f'{eval_dir}/input_{counter * 4 + i}.tiff', input)
        counter += 1
    
    length = counter * 4
    for metric in metrics.keys():
        print(metric.upper(), ':')
        for model in metrics[metric].keys():
            print(f"\t{model.upper()}: {metrics[metric][model]/length:.4f}")




if __name__ == '__main__':
    main()
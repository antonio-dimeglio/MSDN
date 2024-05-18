# MSDN

A reimplementation of the Mixed-Scale Dense Neural Network (MSDNet) in PyTorch.

## Usage
The conda environent can be created using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Which creates an environment named `msdn`. To activate the environment, run:

```bash
conda activate msdn
```

To generate a dataset, first run the phantom generation script:

```bash
python data_generation/phantom_generation.py
```

Then, run the sinogram generation script:

```bash
python data_generation/sinogram_generation.py
```

Finally, run the training script:

```bash
python train.py
```
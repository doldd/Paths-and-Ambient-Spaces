# Paths-and-Ambient-Spaces


## Description

This project is the source of the paper "Paths and Ambient Spaces in Neural Loss Landscapes," which analyzes various aspects of paths and ambient spaces in the loss landscape of neural networks. This repository includes several Jupyter notebooks and Python scripts for different experiments and analyses. All experiments were logged using a local Weights & Biases (wandb) instance, and log files can be provided upon request.

## Notable Files and Directories

- `configs/`: Directory containing configuration files for wandb sweeps.
- `figures/`: Directory for storing generated figures.
- `src/`: Source code directory containing utility functions and other scripts.
- `proba_sandbox/`: Directory for initializing the UCI dataset.
- `pyproject.toml`: Project configuration file for Poetry.
- `README.md`: This file.

## Notebooks
The notebooks contain the analysis of the hyperparameter grid stored with Weights & Biases (wandb) and some single experiments.

- `bezier_length_dnn.ipynb`: Analyse the Curve length.
- `elu_relu_tanh.ipynb`: Compares different activation functions.
- `full_space_sampling.ipynb`: Toy example of full space sampling.
- `functional_divercity.ipynb`: Inspect functional diversity for different "subspace (K)"-dimensions.
- `grid_activation_smoothness.ipynb`: Inspect the subspace smoothness of different activaton functions.
- `mnist.ipynb`: Jupyter notebook to analyse th MNIST dataset experiments.
- `RMF_Frame_creation.ipynb`: Notebook for RMF frame creation.
- `space_temperature_sampler_compare.ipynb`: Inspect the logged wandb results on the synthetic dataset.
- `subspace_sampling_example.ipynb`: An example notebook demonstrating how to optimize a path and sample using the tunnel lifting approach with BlackJAX on a synthetic dataset.

## Scripts
Multiple scripts which are used to run the hyperparameter grid defined in the configs/ directory.

- `jax_run_mnist.py`: Script for running MNIST experiments.
- `jax_sub_path_optim.py`: Script for path optimization used by wandb sweep.
- `jax_sub_sampling_from_path.py`: Script for sampling from optimized path used by wandb sweep.
- `bezier_lengt_dnn_grid.py`: Script to run experiments analyzing length behavior in a long-term environment.

## Dependencies

The project uses Poetry for dependency management. Key dependencies include:
- JAX
- Flax
- WandB
- NumPy
- Pandas
- Matplotlib
- Seaborn
- ArviZ
- BlackJAX

## Getting Started

To get started with this project, clone the repository and install the dependencies using Poetry:

```sh
git clone https://github.com/doldd/Paths-and-Ambient-Spaces.git
cd Paths-and-Ambient-Spaces
poetry install
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
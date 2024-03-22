# Sorting Out Transformers

This repository contains the code, models, and analysis for the "Sorting Out Transformers" project. The project aims to investigate the effects of weight decay on transformer models and perform various interpretations and analyses.

## Requirements

To set up the correct environment and install the necessary dependencies, please refer to the `requirements.txt` file. You can create a new virtual environment and install the required packages by running:

```
pip install -r requirements.txt
```

## Repository Structure

The repository is organized as follows:

- `model/`: This directory contains fully trained transformer models with different levels of weight decay:
  - `sort_no_decay_l1_activation.pth`: Model trained without weight decay.
  - `sort_mid_decay_l1_activation.pth`: Model trained with medium weight decay.
  - `sort_high_decay_l1_activation.pth`: Model trained with high weight decay.

- `checkpoints/`: This directory contains checkpoints of the models after each training epoch. The checkpoints are organized by weight decay level and epoch number.

- `logs.md`: This file contains thoughts, plans, and notes related to the project.

- `analysis.ipynb`: This Jupyter notebook contains the code for training the transformer models and performing preliminary analysis on the trained models.

- `patching.ipynb`: This Jupyter notebook contains the code for activation patching, query-key (QK) analysis, output-value (OV) circuit analysis, and other interpretation attempts on the trained models.

## Usage

To reproduce the results or explore the project further, follow these steps:

1. Set up the environment by installing the required packages from `requirements.txt`.

2. Run the `analysis.ipynb` notebook to train the transformer models with different levels of weight decay and perform preliminary analysis.

3. Run the `patching.ipynb` notebook to apply activation patching, conduct QK and OV circuit analysis, and attempt other interpretation techniques on the trained models.

4. Run `utils/llc.py` to generate the LLC data (This will take a long time).

Feel free to explore the code, modify the notebooks, and experiment with different configurations to gain further insights into the effects of weight decay on transformer models and their interpretability.

## Acknowledgments

We would like to thank the authors of the original transformer paper and the developers of the libraries and frameworks used in this project for their valuable contributions to the field of natural language processing.

If you have any questions or suggestions, please feel free to open an issue or contact the project maintainers.

Happy exploring!
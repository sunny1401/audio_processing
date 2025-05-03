# Training Template
- This repository provides a minimal template for optimized training of deep learning models with reproducible results. 
- It provides configurable code for training using pytorch-lightning for both Slurm and Single GPU.
- The repository has code for:
 - PyTorch Lightning Trainer and Pipeline 
 - HPO Pipeline Trainer using Ray-Tune  
 - Wandb Logger 
 - LR Scheduler to be used along with Trainer; which includes [LARS LR Scheduler from MAE](https://github.com/facebookresearch/mae/blob/main/util/lars.py)
 - Feature Extraction Helper Base class to extract features from different models
 - Downstream Logistic Regression based Classification and Segmentation Tasks, for easy comparison of learned features without additional processing.

## Customization
To customize the current repo, template the repo to create a new repo. After the repo is created, the package folder needs to be updated if you want to customize and change. This needs to be done as follows:

- in pyproject.toml, update the lines 1 -7 as needed
- Add python packages as needed


```bash
pip install poetry
poetry install
```
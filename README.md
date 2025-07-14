# Deployment Environment Setup

This directory contains a subset of the main voxel-diffusion repository configured for deployment purposes.

## Environment Setup

### 1. Create the conda/micromamba environment

```bash
micromamba env create -f deploy-environment.yml
micromamba activate voxel-diffusion-deploy
```

### 2. Manual Dependencies

FVDB needs to be built manually as it's not available through conda/pip:

Follow the official FVDB installation instructions. Make sure to use the feature branch and build with CUDA support. 

This typically involves cloning the repo and building from source with this 
environment active. 


### 3. Verify Installation

After setting up the environment and building FVDB, verify that all components work:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import fvdb; print('FVDB successfully imported')"
python deploy_imports.py
```

## Notes

- PyTorch version is 2.7.0
- CUDA 12.0 to 12.9 is supported
- The environment name is `voxel-diffusion-deploy`

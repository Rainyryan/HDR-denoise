# HDR Denoising and Distillation Repository

This repository contains PyTorch code for high dynamic range (HDR) image denoising and model distillation research. It is based on Yiheng Chi's 2023 CVPR paper, *HDR Imaging with Spatially Varying Signal-to-Noise Ratios*. The main goal is to reduce the model size by 100× while maintaining the same performance, and to push for better generalizability through lightweight student distillation and noise-aware HDR restoration. The code includes training and evaluation scripts for HDR restoration models and lightweight student-distillation experiments.

## Key Features

- HDR denoising model training using custom HDR image dataset pipelines.
- Knowledge distillation from a larger HDR teacher model into a smaller student model.
- Support for HDR data augmentation, photon noise simulation, and expanded dynamic-range training.
- Key architecture change: transformer model backbone swapped from SwinIR to Restormer.
- Multiple model architectures and experimental variants under `HDR_model.py` and `HDR_model_switch.py`.

## Repository Structure

- `train_HDR.py` - Main HDR model training script.
- `distill_train_HDR.py` - Distillation training script for teacher/student models.
- `CoDe_test_script.py` - Example script that tests the content-decoupling module.
- `HDR_dataset.py` - Dataset loader, HDR utilities, and noise generation functions.
- `HDR_model.py` and `HDR_model_switch.py` - Model definitions and architecture variants.
- `my_GBTF.py` - CFA interpolation helper used by the dataset pipeline.
- `requirements.txt` - Python dependencies.
- `models_*` and `distill_models_*` directories - Saved checkpoints from training experiments.
- `results_*` directories - Evaluation outputs and experiment results.
- `test_HDR_*` scripts - Test cases for research components.

## Requirements

- Python 3.8+ recommended
- PyTorch
- torchvision
- numpy
- opencv-python
- lpips

Install dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Setup

1. Clone the repository.
2. Install dependencies.
3. Prepare your HDR dataset.

### Dataset

This project expects an HDR dataset directory containing image files suitable for HDR restoration training. Current training scripts use a hard-coded dataset path such as:

- `train_HDR.py`: `/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/train`
- `distill_train_HDR.py`: `/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/train`

Update the dataset path in the script before training if your data is stored elsewhere.

## Running Training

### Train an HDR model

The default script is `train_HDR.py`.

```bash
python train_HDR.py
```

Modify the top section of the file to configure:

- `save_folder`
- `pretrained_path`
- learning rate and optimizer
- batch size and epoch count
- dataset path

### Train a distilled student model

Use `distill_train_HDR.py` to train a smaller student model guided by a larger teacher model.

```bash
python distill_train_HDR.py
```

Update the script to point to a teacher checkpoint and set your desired distillation experiment parameters.

## Testing and Visualization

There are several research test scripts in the repository:

- `test_HDR_all_overlap.py`
- `test_HDR_attention_map.py`
- `test_HDR_preexpand_all_distill.py`
- `test_HDR_preexpand_all_overlap.py`
- `test_HDR_transposed_attention.py`

Run these directly with Python as needed to verify model components.

## Notes

- This repository is primarily research code, so most scripts are configured as self-contained Python files rather than CLI tools.
- The dataset and checkpoint paths are currently hard-coded; edit the paths in scripts before execution for your local environment.
- HDR dataset preparation uses custom noise models and expand operations implemented in `HDR_dataset.py`.

## Contact

For questions or assistance with this project, inspect the training scripts and model files to adapt hyperparameters and dataset handling for your experiments.

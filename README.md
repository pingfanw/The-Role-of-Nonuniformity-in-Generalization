# Numerical Experiments of Paper "Beyond Sharpness: The Role of Nonuniformity in Generalization".
This implementation includes consistent sharpness-aware minimization and kroncker factor approximate variance algroithms across Imagenet, CIFAR10/100, Mnist and food101 datasets.

## 1. Motivation and Background

Understanding why deep neural networks generalize well, even when trained to near-zero training error, remains a fundamental open question in deep learning theory.
A traditional perspective attributes good generalization to **flat minima** or **low sharpness** in the loss landscape, implying that parameter perturbations around the solution cause only small loss increases.

However, recent studies indicate that sharpness alone is insufficient to explain modern networks’ generalization behaviors:

* Flat minima can still generalize poorly in certain regimes.
* The empirical correlation between sharpness and test error is inconsistent across architectures, datasets, and optimizers.

This project explores a complementary dimension—**nonuniformity**—as a key factor in generalization.
Here, *nonuniformity* broadly refers to the uneven distribution of important quantities (e.g., gradients, parameter updates, per-sample losses, or class-wise statistics) during optimization.
The accompanying paper hypothesizes that lower nonuniformity in the training dynamics correlates with better generalization, even when sharpness levels remain similar.


## 2. Repository Structure

* **`main.py`** – Entry point for running experiments (datasets, models, optimizers, measurement options).
* **`models/`** – Neural network architectures such as *ResNet*, *Vision Transformer*, etc.
* **`optimizer/`** – Optimizers including *SGD*, *Adam*, *SAM (Sharpness-Aware Minimization)*, and the proposed *Nonuniformity-Aware* variants.
* **`utils/`** – Utility scripts for dataset loading, metric computation, visualization, and logging.


## 3. Installation

### Requirements

Use **Python ≥ 3.8** and install the following core dependencies:

```bash
pip install torch torchvision tqdm numpy matplotlib
```

For mixed precision or distributed training, you may additionally require NVIDIA Apex or `torch.distributed`.


## 4. Running Experiments

Example command:

```bash
python main.py \
  --dataset CIFAR10 \
  --model ResNet18 \
  --optimizer SAM \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1 \
  --measure_nonuniformity
```

### Main Arguments

| Argument                           | Description                                                               |
| ---------------------------------- | ------------------------------------------------------------------------- |
| `--dataset`                        | Dataset name: `MNIST`, `CIFAR10`, `CIFAR100`, `ImageNet`, `Food101`, etc. |
| `--model`                          | Model architecture (e.g., `ResNet18`, `ViT`, …).                          |
| `--optimizer`                      | Optimizer type: `SGD`, `Adam`, `SAM`, `NonuniformSAM`.                    |
| `--epochs`, `--batch_size`, `--lr` | Standard training hyperparameters.                                        |
| `--measure_nonuniformity`          | Enables computation and logging of nonuniformity metrics.                 |
| `--save_dir`                       | Directory for saving logs and checkpoints.                                |


## 5. Metrics and Core Concepts

### **Sharpness**

Measures the local sensitivity of the loss landscape around the learned parameters. Implementations include SAM-style perturbation and spectral-norm approximations.

### **Nonuniformity**

Quantifies the unevenness of parameter updates, gradient magnitudes, or sample-wise losses.
Examples:

* Variance of per-channel gradient norms.
* Skewness/kurtosis of update magnitudes across layers.
* Class-wise loss imbalance or feature activation disparity.

### **Variance and Correlation**

Tracks parameter or gradient variance across iterations and its correlation with test error.
Together with nonuniformity, these quantities reveal how optimization dynamics shape generalization.


## 6. Expected Observations

Empirical findings (consistent with the associated paper):

* Sharpness correlates with generalization **only partially**.
* Nonuniformity exhibits a stronger, more consistent correlation with generalization error.
* Models with *lower nonuniformity* generalize better even at similar sharpness levels.
* Optimizers that implicitly reduce nonuniformity (e.g., SAM, adaptive SAM variants) yield improved test accuracy.


## 7. Visualization and Logging

Results are automatically logged (e.g., using `tqdm`, `tensorboardX`, or `matplotlib`).
Typical outputs include:

* Training/Test accuracy and loss curves.
* Nonuniformity and sharpness metrics per epoch.
* Comparison plots across optimizers and datasets.
* Saved checkpoints and summary tables under `save_dir/`.

Example visualization (to be generated with `utils/plot_utils.py`):

```bash
python utils/plot_utils.py --logdir ./exp_nonuniformity
```


## 8. Citation

If you use or build upon this repository, please cite:

```bibtex
@article{2025nonuniformity,
  title   = {Beyond Sharpness: The Role of Nonuniformity in Generalization},
  author  = {Yingcong Zhou, Pingfan Wu, Li Wang, Zhiguo Fu, Fengqin Yang},
  journal = {Proceedings of AAAI 2025},
  year    = {2025}
}
```


## 9. License

This repository is released under the **MIT License**.

<div align="center">
# Potato Leaf Disease & Nematode Symptom Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ILuuI/potato-leaf-disease-classifier/blob/main/notebooks/efficientnetb0_potato_leaf_sorter.ipynb)

Deep learning system for detecting fungal diseases and nematode-related leaf symptoms in potato (*Solanum tuberosum L.*) crops using **EfficientNet-B0** and transfer learning. Developed as part of an Electronic Engineering research project at Universidad de Sucre (Colombia), aimed at providing an accessible, low-cost diagnostic tool for small potato and root-crop producers.
</div>

## Overview

Potato is one of the most important food crops worldwide, but yields are heavily impacted by fungal diseases and parasitic nematodes. Visual inspection by experts remains the standard diagnostic method, which is slow, subjective, and largely inaccessible in rural areas with limited technical assistance.

This project trains and evaluates CNN-based image classifiers to automatically distinguish between:

| Class | Label |
|---|---|
| 0 | Healthy leaf |
| 1 | Late blight (*Phytophthora infestans*) |
| 2 | Nematode-associated leaf symptoms (*Globodera pallida*) |
| 3 | Early blight (*Alternaria solani*) |

Two parallel experiments were run on identical infrastructure to isolate the effect of the severely imbalanced nematode class:

- **Experiment A** — 4 classes (Healthy, Late blight, Nematodes, Early blight)
- **Experiment B** — 3 classes (same split, nematode class removed), used as a controlled baseline

## Dataset

Images were aggregated from four public, verifiable sources:

| Source | Contributes to | 
|---|---|
| [Multi-Crop Leaf Disease Dataset](https://data.mendeley.com/datasets/z6jp232g5j/1) (Mendeley Data) | Nematodes (all), partial Healthy & Late blight |
| [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (Kaggle) | Healthy, Late blight, Early blight |
| [Potato Leaf (Healthy and Late Blight)](https://www.kaggle.com/datasets/nirmalsankalana/potato-leaf-healthy-and-late-blight) (Kaggle) | Healthy, Late blight |
| [Potato Computer Vision Model](https://universe.roboflow.com/ai-xwoe2/potato-eq4aq) (Roboflow Universe) | Healthy, Late blight, Early blight |

The nematode class originally had only 68 images in the Multi-Crop dataset; several photos contained more than one leaf, so each leaf was manually cropped out individually, yielding 90 usable images. For the other three classes, images from the four sources were merged and deduplicated (matching filenames and `_aug`/`aug_` suffixes; only one image kept per group of augmented variants sharing a base index).

**Final image count per class** (after deduplication, before balancing):

| Class | Images |
|---|---|
| Healthy | 1,560 |
| Late blight | 2,593 |
| Nematodes | 90 |
| Early blight | 2,303 |

> **Note on reproducibility:** an earlier iteration of this project used an additional data source for the healthy/late blight/early blight classes that could not be identified with certainty after the fact. To guarantee full traceability, the system was retrained end-to-end using only the four verified sources listed above. All results below correspond to that retraining.

## Results

Both experiments follow a two-phase transfer learning protocol: 30 epochs with a frozen EfficientNet-B0 backbone, followed by 30 epochs of full fine-tuning at a reduced learning rate.

### Experiment A — 4 classes

| Phase | Global Accuracy |
|---|---|
| 1 (frozen backbone) | 94.83% |
| 2 (fine-tuned) | **99.07%** |

| Class | Precision (P2) | Recall (P2) | F1 (P2) |
|---|---|---|---|
| Healthy | 0.9869 | 1.0000 | 0.9934 |
| Late blight | 1.0000 | 0.9735 | 0.9865 |
| Nematodes | 1.0000 | 1.0000 | 1.0000 |
| Early blight | 0.9825 | 0.9956 | 0.9890 |

### Experiment B — 3 classes (control)

| Phase | Global Accuracy |
|---|---|
| 1 (frozen backbone) | 94.40% |
| 2 (fine-tuned) | **98.82%** |

| Class | Precision (P2) | Recall (P2) | F1 (P2) |
|---|---|---|---|
| Healthy | 0.9869 | 1.0000 | 0.9934 |
| Late blight | 0.9825 | 0.9912 | 0.9868 |
| Early blight | 0.9955 | 0.9735 | 0.9843 |

**Key finding:** despite the nematode class having only 90 original images (vs. 1,500+ for the other classes after balancing), incorporating it — via targeted augmentation and class-weighted loss — did not meaningfully degrade performance on the majority classes. Global accuracy between the 4-class and 3-class experiments differs by 0.43 points in Phase 1 and 0.25 points in Phase 2, and the nematode class itself reached a perfect 1.0000 precision/recall after fine-tuning.

Full metrics, confusion matrices, and training curves are available in the [research paper](docs/research_paper.pdf) and reproduced in the notebook.

## Methodology summary

- **Data balancing:** targeted data augmentation raised the nematode class from 90 → 500 images (rotation, brightness/contrast jitter, mild Gaussian blur), while the majority classes were undersampled to 1,500 images each.
- **Preprocessing:** symmetric black padding to square aspect ratio + LANCZOS resize to 224×224, to avoid distorting natural leaf proportions — especially important given the manually cropped nematode images, which retained highly heterogeneous aspect ratios.
- **Split:** stratified 70/15/15 train/val/test (fixed seed for reproducibility); Experiment B reuses the exact same split as Experiment A minus the nematode class, for a controlled comparison.
- **Training augmentation:** horizontal/vertical flips, random rotation, color jitter, random translation, ImageNet normalization.
- **Architecture:** EfficientNet-B0 (ImageNet-pretrained), custom classifier head — `Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→N)`.
- **Training protocol:** Adam optimizer, `ReduceLROnPlateau` scheduler, class-weighted `CrossEntropyLoss` for the 4-class experiment to penalize nematode errors more heavily.

EfficientNet-B0 was selected over VGG16, ViT, and MobileNetV2 for its accuracy-to-size ratio (~20MB), making it viable for future deployment on mobile/edge devices in the field.

## Repository structure

```
potato-leaf-disease-classifier/
├── notebooks/
│   └── efficientnetb0_potato_leaf_sorter.ipynb   # Full training & evaluation pipeline
├── docs/
│   └── research_paper.pdf                        # Full research write-up (Spanish)
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting started

The notebook was developed and run on **Google Colab** (T4 GPU) with the dataset stored on Google Drive. To reproduce:

1. Open `notebooks/efficientnetb0_potato_leaf_sorter.ipynb` in Google Colab.
2. Mount your Google Drive and update `BASE_DIR` to point to your dataset location.
3. Download and merge the four public sources listed in [Dataset](#dataset) above into four class folders (`0` healthy, `1` late blight, `2` nematodes, `3` early blight), following the deduplication approach described in the paper.
4. Run cells sequentially — sections are numbered and documented (dependencies → preprocessing → balancing → training Phase 1/2 → evaluation → confusion matrices → sample predictions).

To run locally instead of Colab, install the dependencies:

```bash
pip install -r requirements.txt
```

Note: Colab-specific cells (`google.colab.drive`, `!pip install` magics) will need to be adapted for a local environment.

## Limitations & future work

- The nematode class's 500 training images derive from augmentation of only 90 manually-cropped photos (originally 68 multi-leaf images), so they are not fully statistically independent — the model may have learned artifacts specific to that small source set. Robust validation would require nematode images captured under different field conditions.
- Images were sourced from public datasets with natural background/lighting variability rather than a controlled lab setting, which is reflected in the reported metrics being slightly below (but comparable to) lab-condition benchmarks such as PlantVillage-only studies.
- Planned next steps: expanding the nematode image bank through collaboration with regional agricultural institutions, testing larger architectures (EfficientNet-B4, Vision Transformers), and building a mobile deployment interface for field use by small producers.

## Authors

- Jesús Escudero Álvarez — Electronic Engineering, Universidad de Sucre
- Lucas Rivadeneira Zarza — Electronic Engineering, Universidad de Sucre

## Citation

If you reference this work, please cite the accompanying paper:

> J. Escudero Álvarez and L. Rivadeneira Zarza, "Sistema de Detección de Hongos y Síntomas de Nemátodos en Hojas de Papa Mediante Visión Artificial y Aprendizaje Profundo," Universidad de Sucre, 2026.

## License

This project is released under the [MIT License](LICENSE). The research paper (`docs/research_paper.pdf`) is included for reference; check with the authors before reuse of the written content itself.

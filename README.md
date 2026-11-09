
# Malaria Parasite Detection: Classical Computer Vision vs Deep Learning

Classifying red blood cell microscope images as **parasitised** or **uninfected** with malaria. The project compares a hand-built computer vision pipeline against a convolutional neural network, under an evaluation protocol designed to avoid the data leakage common in public solutions to this dataset.

> **In progress.** The evaluation framework and the first classical model are working. The CNN and the full feature set are under development.

![Python](https://img.shields.io/badge/Python-3.13-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-planned-lightgrey)

---

## Why this project

Malaria is diagnosed by a trained expert counting parasites under a microscope, which is slow and hard to scale. Automated cell screening could help, but only if it is **sensitive**: a missed infection is far worse than a false alarm. So this project reports **recall** on infected cells alongside accuracy.

It also asks a practical question: **how close can interpretable, hand-crafted features get to a CNN, and what does each approach actually look at?**

## Highlights

- **Leakage-free evaluation.** The dataset contains many cells from each patient. Many public notebooks split images at random, so cells from the same patient end up in both training and test sets and the scores are inflated. This project uses **patient-grouped stratified 5-fold cross-validation**, with hyperparameter tuning also grouped by patient in the inner loop.
- **One reusable experiment harness.** Every model (SVM, kNN, and later the CNN and a hybrid) plugs into the same `run_cv()` function. Metrics are logged automatically to CSV, and the report tables are generated from those files rather than typed by hand.
- **Hand-crafted features computed only on the cell.** Colour statistics are computed inside a segmented cell mask, so the black background does not dominate the features.
- **Tested, cached pipeline.** Each feature group is cached once as a NumPy array in a fixed row order. Classifier experiments then take seconds to minutes instead of re-processing 27k images each time. The core modules have unit tests (pytest).

## Early results

Patient-level 5-fold cross-validation, **colour features only**, default hyperparameters (no tuning yet):

| Model | Accuracy | Recall (infected) | ROC-AUC |
|---|---|---|---|
| kNN (k = 5) | 84.8 ± 1.6% | 77.4% | 0.918 |
| SVM (RBF kernel) | **93.2 ± 1.4%** | **92.0%** | **0.978** |

A single feature group already reaches 93% accuracy without leaking patients between folds. These are baselines: the remaining feature groups, tuning and the CNN comparison are still to come.

## Approach

```
cell image ─► background mask ─► hand-crafted features ─► SVM / kNN      (System A: classical)
          └─► ResNet-18 fine-tuned on cells ───────────────────────────► (System B: deep)
              CNN embedding + hand-crafted features ─► SVM ────────────► (System C: hybrid)
```

**Classical pipeline**
| Stage | Techniques | Status |
|---|---|---|
| Cell masking | grey-level thresholding, hole filling | ✅ |
| Colour features | 512-bin HSV histogram + mean/std/skew of 6 RGB/HSV channels (530 values) | ✅ |
| Classifiers | RBF-kernel SVM, kNN, standardisation inside the pipeline, grouped grid search | ✅ |
| Parasite segmentation | CLAHE, filtering, Otsu thresholding, morphology, connected components | 🔄 |
| Shape and region features | Hu moments, circularity, solidity, Fourier descriptors, blob statistics | 🔄 |
| Interest-point features | Difference-of-Gaussians blobs, dense SIFT + Bag of Visual Words | 🔄 |

**Planned experiments:** feature ablation (which features matter), data efficiency (performance with 1% to 100% of the training data), robustness to blur, noise and stain colour shifts, and **Grad-CAM** heat-maps compared with the classical parasite masks, to check whether the CNN looks where the parasites are.

## Tech stack

Python · OpenCV · scikit-image · scikit-learn · NumPy · pandas · SciPy · pytest · PyTorch / torchvision (CNN, planned) · Google Colab (GPU training)

## Project structure

```
src/
├── make_splits.py        # patient-grouped 5-fold split → splits.csv
├── data.py               # image loading
├── evaluate.py           # metrics → results/*.csv, summary and LaTeX tables
├── run_cv.py             # cross-validation harness shared by every model
├── features/colour.py    # HSV histogram + colour moments inside the cell mask
├── build_features.py     # computes and caches features to .npy
├── classical_utils.py    # feature loading, grouped grid search
├── svm.py, knn.py        # classical classifiers
└── cnn.py                # ResNet-18, embeddings, Grad-CAM (in progress)
tests/                    # pytest unit tests
results/                  # one CSV per experiment
```

## Running it

```bash
git clone <repo-url> && cd <repo-folder>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download the dataset (needs a Kaggle API token)
bash scripts/download_data.sh

pytest -q                         # run the tests
python -m src.build_features      # compute and cache features
python -m src.svm                 # 5-fold patient-level CV → results/main.csv
python -m src.knn
```

## Roadmap

- [x] Patient-grouped evaluation harness and metrics logging
- [x] Colour features, SVM and kNN baselines
- [ ] Otsu / morphology parasite segmentation
- [ ] Shape, region, DoG and SIFT Bag-of-Visual-Words features
- [ ] Tuned classical model on all features, with a feature ablation
- [ ] Fine-tuned ResNet-18 and a hybrid CNN + hand-crafted model
- [ ] Grad-CAM interpretability, robustness and data-efficiency experiments
- [ ] Written report (CVPR format)

## Dataset and references

- [NIH "Cell Images for Detecting Malaria"](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria): 27,558 cell images from Giemsa-stained thin blood smears, balanced between the two classes.
- S. Rajaraman et al. *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ, 2018.

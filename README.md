# ISIC 2024 – Multimodal Skin Cancer Detection

A multimodal machine learning pipeline developed for the **ISIC 2024 – Skin Cancer Detection with 3D-TBP** Kaggle competition. The project combines image-based deep learning models and patient metadata through a stacking ensemble to improve melanoma detection performance.

The proposed solution integrates two convolutional neural networks (**ResNet18** and **EfficientNet-B0**) with a **CatBoost** classifier trained on tabular metadata. Their predictions are combined using a **Logistic Regression** meta-model, producing a unified malignancy probability for each lesion.

The repository contains the complete training and inference pipeline, including cross-validation, final model training, multimodal stacking, and submission generation, providing a fully reproducible workflow from raw competition data to the final Kaggle submission.

## Project Highlights

* **Multimodal classification pipeline** combining dermoscopic images and patient metadata.
* **Dual image models** based on **ResNet18** and **EfficientNet-B0**.
* **CatBoost metadata classifier** trained on the complete ISIC 2024 metadata.
* **Logistic Regression stacking** to combine image and metadata predictions into a single malignancy score.
* **Five-fold Stratified Group Cross-Validation** for robust model evaluation and unbiased Out-of-Fold predictions.
* **Official ISIC 2024 evaluation metric (competition pAUC)** used throughout training and validation.
* **Fully reproducible workflow**, including model training, stacking, and submission generation.

## Pipeline Overview

The proposed solution follows a multimodal machine learning pipeline that combines predictions from metadata-based and image-based models through a stacking ensemble.

The workflow begins by training three independent base models:

* **CatBoost** on patient metadata and lesion attributes.
* **ResNet18** on dermoscopic images.
* **EfficientNet-B0** on dermoscopic images.

Each model is first evaluated using five-fold Stratified Group Cross-Validation to generate unbiased Out-of-Fold predictions. After validating the training strategy, the selected models are retrained on the complete training dataset.

Finally, the predictions produced by the three base models are standardized and combined using a **Logistic Regression** meta-model, generating the final malignancy probability used for submission.

The complete workflow is illustrated below.

<p align="center">
  <img src="figures/pipeline_overview.png" alt="Pipeline Overview" width="900">
</p>

## Repository Structure

The repository is organized as a sequence of independent notebooks, each responsible for a specific stage of the machine learning pipeline. This modular design improves readability, simplifies experimentation, and allows every step—from model development to final inference—to be reproduced independently.

The overall repository organization is illustrated below.

<p align="center">
  <img src="figures/repository_structure.png" alt="Repository Structure" width="900">
</p>

The main notebooks are organized as follows:

| Notebook                                     | Description                                                                                                                                         |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01_catboost_5fold_cv.ipynb**               | Trains and evaluates the metadata model using Stratified Group Cross-Validation and generates Out-of-Fold predictions.                              |
| **02_catboost_final_training.ipynb**         | Retrains the CatBoost model on the complete metadata dataset for final inference.                                                                   |
| **03_image_5fold_cv.ipynb**                  | Trains and evaluates the ResNet18 and EfficientNet-B0 image classifiers using five-fold cross-validation.                                           |
| **04_image_final_training.ipynb**            | Retrains the selected image architectures on the complete training dataset.                                                                         |
| **05_logistic_regression_stacking.ipynb**    | Trains the meta-model using Out-of-Fold predictions from the three base models and compares the stacking strategy with simpler ensemble approaches. |
| **06_final_submission.ipynb**                | Loads the pretrained models, performs inference on the competition test set, and generates the final Kaggle submission file.                        |

## Methodology

The proposed solution adopts a multimodal ensemble strategy that combines complementary sources of information available in the ISIC 2024 dataset. Independent models are trained on patient metadata and dermoscopic images, and their predictions are subsequently combined through a Logistic Regression stacking model.

Each component of the pipeline is developed and evaluated independently before being integrated into the final ensemble. This modular approach simplifies experimentation while allowing each model to exploit the strengths of its respective data modality.

The methodology is composed of three main stages:

1. Metadata classification using CatBoost.
2. Image classification using convolutional neural networks.
3. Multimodal prediction fusion through Logistic Regression stacking.

### Metadata Model

Patient metadata and lesion descriptors are modeled using **CatBoost**, a gradient boosting algorithm particularly well suited for heterogeneous tabular datasets containing both numerical and categorical variables.

The metadata preprocessing pipeline includes feature engineering, categorical feature handling, and the generation of Out-of-Fold predictions through five-fold Stratified Group Cross-Validation. After validating the selected hyperparameters, a final CatBoost model is retrained using the complete metadata dataset and later used during inference.

### Image Models

Dermoscopic images are modeled using two complementary convolutional neural network architectures: **ResNet18** and **EfficientNet-B0**. Both models are initialized with ImageNet pretrained weights and fine-tuned on the ISIC 2024 training dataset.

Model development follows a two-stage training strategy. First, each architecture is evaluated using five-fold Stratified Group Cross-Validation to generate unbiased Out-of-Fold predictions and estimate its generalization performance. Once the training configuration is validated, the selected architectures are retrained on the complete dataset to produce the final models used during inference.

The resulting image classifiers capture visual characteristics of skin lesions that complement the information available in the patient metadata.

### Multimodal Stacking

The final prediction is obtained by combining the outputs of the three base models through a **Logistic Regression** stacking ensemble.

Out-of-Fold predictions generated during cross-validation are used as training data for the meta-model, ensuring that the stacking model is trained exclusively on predictions produced from unseen samples. This prevents information leakage while allowing the Logistic Regression classifier to learn the optimal contribution of each base model.

During inference, prediction probabilities from the final CatBoost, ResNet18, and EfficientNet-B0 models are passed to the trained Logistic Regression model, which produces the final malignancy probability submitted to the competition.

This multimodal strategy consistently outperformed each individual model as well as a simple weighted-average ensemble, demonstrating the benefit of learning an optimal combination of metadata-based and image-based predictions. TODO: Continue the readme from here

---
## Project Overview

Early detection of skin cancer is critical for improving patient outcomes. The ISIC 2024 challenge focuses on predicting whether a skin lesion is malignant by leveraging dermoscopic images and patient metadata.

This repository presents a multimodal ensemble approach composed of:

* **ResNet18** trained on dermoscopic images.
* **EfficientNet-B0** trained on dermoscopic images.
* **CatBoost** trained on patient and lesion metadata.
* **Logistic Regression** stacking to combine the predictions of the three base models.

The project was implemented using **PyTorch**, **CatBoost**, and **scikit-learn**.

---

## Features

- Metadata classification using CatBoost
- Image classification using ResNet18
- Image classification using EfficientNet-B0
- Five-fold Stratified Group Cross Validation
- Logistic Regression stacking
- Fully reproducible training and inference pipeline
- Modular notebook organization

---

## Pipeline

<img width="1310" height="1138" alt="pipeline" src="https://github.com/user-attachments/assets/78101c78-4903-4cb8-a9ca-0ca2d4963bad" />


## Repository Structure

<img width="624" height="1024" alt="structure" src="https://github.com/user-attachments/assets/62aafb1c-438c-4dae-85a6-b525d298b191" />

---

## Project Structure

```text
.
├── notebooks/          # Training and experimentation notebooks
├── src/                # Dataset, models and utility modules
├── figures/            # Figures used in the README and presentation
├── docs/               # Project documentation
├── requirements.txt
└── README.md
```

---

## Methodology

### Image Models

Two convolutional neural networks were trained independently using dermoscopic images:

* ResNet18
* EfficientNet-B0

Both models produce the probability of malignancy for each lesion.

### Metadata Model

A CatBoost classifier was trained using the tabular metadata provided by the competition, including demographic and lesion-related attributes.

### Stacking Ensemble

The final prediction is obtained by combining the outputs of the three base models using a Logistic Regression meta-model trained on out-of-fold predictions.

```text
Dermoscopic Images
        │
 ┌──────┴────────┐
 │               │
ResNet18   EfficientNet-B0
 │               │
 └──────┬────────┘
        │
Patient Metadata
        │
     CatBoost
        │
        ▼
 Three Prediction Scores
        │
        ▼
 Logistic Regression
        │
        ▼
 Final Malignancy Probability
```

---

## Results

| Model                        |   OOF pAUC |
| ---------------------------- | ---------: |
| ResNet18                     |     0.8109 |
| EfficientNet-B0              |     0.8462 |
| CatBoost                     |     0.8609 |
| Weighted Average Ensemble    |     0.8773 |
| Logistic Regression Stacking | **0.8844** |

The stacking approach achieved the best overall performance, outperforming every individual model and the weighted-average ensemble.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/isic2024-skin-cancer-detection.git
cd isic2024-skin-cancer-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is provided by the ISIC 2024 Kaggle competition and is **not** included in this repository.

Competition page:

https://www.kaggle.com/competitions/isic-2024-challenge

---

## Technologies

* Python
* PyTorch
* CatBoost
* scikit-learn
* pandas
* NumPy
* OpenCV
* timm
* Kaggle Notebooks

---

## Future Improvements

Potential directions for future work include:

* Training larger image backbones (EfficientNetV2, ConvNeXt, ViT).
* Advanced test-time augmentation (TTA).
* Additional metadata feature engineering.
* More sophisticated stacking and blending strategies.
* Probability calibration for the ensemble.

---

## License

This project is released for educational and research purposes.

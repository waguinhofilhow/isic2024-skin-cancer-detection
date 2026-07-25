# ISIC 2024 Skin Cancer Detection

An ensemble learning pipeline developed for the **ISIC 2024 - Skin Cancer Detection with 3D Total Body Photography** Kaggle competition. This project combines image-based deep learning models and metadata-based machine learning to improve the detection of malignant skin lesions.

## Project Overview

Early detection of skin cancer is critical for improving patient outcomes. The ISIC 2024 challenge focuses on predicting whether a skin lesion is malignant by leveraging dermoscopic images and patient metadata.

This repository presents a multimodal ensemble approach composed of:

* **ResNet18** trained on dermoscopic images.
* **EfficientNet-B0** trained on dermoscopic images.
* **CatBoost** trained on patient and lesion metadata.
* **Logistic Regression** stacking to combine the predictions of the three base models.

The project was implemented using **PyTorch**, **CatBoost**, and **scikit-learn**.

---

## Repository Structure

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

# Biomass Prediction From Pasture Images  
Multimodal Deep Learning model for predicting pasture biomass from RGB drone images.

This project implements a **fully image-driven regression model** capable of predicting **five biomass components** directly from an image:

- `Dry_Clover_g`
- `Dry_Dead_g`
- `Dry_Green_g`
- `Dry_Total_g`
- `GDM_g`

Although the training process also uses tabular environmental variables (NDVI, height, state, species, month, season), **the final model is applied solely on image embeddings**.  
This makes it suitable for real-world deployment where only the image is available.

---

## Summary

- Uses modern **image embeddings** + **multitask MLP** architectue  
- Trains with **blind experiments**, matching the Kaggle validation pipeline  
- Includes **full training pipeline** (`train.py`)  
- Includes **inference script producing submission files** (`predict.py`)  
- Modularized in `src/` for clarity and maintainability  
- Reproducible, packaged, version-controlled  

---

## Project Structure

```
project/
│
├── README.md
├── requirements.txt
│
├── data/
│ ├── raw/
│ │ ├── train.csv
│ │ └── test.csv
│ └── external/
│ └── embeddings/
│ └── image_embeddings.csv
│
├── src/
│ ├── train/
│ │ ├── train.py ← training pipeline (from notebook)
│ │ └── predict.py ← inference pipeline
│ │
│ ├── models/
│ │ └── model.py ← ImageOnlyMultitaskNet
│ │
│ ├── data/
│ │ └── dataset.py ← PastureImageOnlyDataset
│ │
│ └── utils/
│ └── misc.py ← rmse, weighted-R², helper functions
│
└── models/
└── weights/
└── image_only_ckpt.pt ← saved model checkpoint
```



---

# Getting Started :D

## 1. Install dependencies

```
pip install -r requirements.txt
```

- Minimum requirements:

```
torch
pandas
numpy
scikit-learn
tqdm
```


## Training

The full training pipeline reproduces exactly the logic of the original notebook (see folder notebooks):

- Builds the wide table
- Merges image embeddings
- Encodes categorical features
- Runs blind experiments
- Calculates global metrics

Saves:

- validation metrics
- blind test predictions
- model checkpoint

To start training:
```
python -m src.train.train
```

The script will:

- load data/raw/train.csv
- load data/external/embeddings/image_embeddings.csv
- build the training dataset
- run 10 blind-fold validation cycles

save outputs in:

```
outputs/blind_true_vs_pred_image_only.csv
models/weights/image_only_ckpt.pt
```

## Inference/Prediction

After training, run:

```
python -m src.train.predict
```

This script will:

- load model checkpoint
- read test.csv
- merge embeddings
- predict the 5 biomass targets for each image

assemble the submission file in:
```
outputs/submission.csv
```


## Model architecture

```
Image Embeddings  →  MLP backbone  →  5-output regression head (biomass)
                                   →  auxiliary numeric head (NDVI, height)
                                   →  auxiliary categorical heads (month, season, state, species)

```

-> Training uses only the biomass loss (alpha_aux = 0), matching the notebook.
-> Auxiliary heads exist only to keep the architecture flexible for future use.
-> Final inference uses only the image embedding → biomass regression.


## Metrics

The scripts compute:
- RMSE
- R²
- Weighted R² 

The blind experiment evaluation aggregates performance over multiple runs.


## Notebooks

Reference notebooks used during development:

- 01_EDA.ipynb
- 02_featureFusion_and_preModeling.ipynb
- 03_multimodal_biomass_model.ipynb
- 04_imageOnly_biomassModel.ipynb (source of the training code)

Notebooks remain in the repo for exploration and documentation only —
all production code lives under src/.


## This repository highlights:

- Deep Learning (PyTorch)

A custom architecture with multitask heads and flexible loss weighting.

-  Multimodal fusion

Image embeddings + tabular variables (used during training).

-  Clean modular code

Training, dataset handling, and models split into logical modules.

- Reproducibility

Exact matching between notebook → script → inference.

- Real ML pipeline

Blind-validation, checkpointing, inference layer, and final submission generation.

- GitHub-ready project

Professional structure, versioned, and easy for recruiters to navigate.


## Author

Marcelo Ciani Vicentin
PhD candidate in Astronomy | USP – University of São Paulo

# Machine Learning and UAS for Managing Autumn Olive on Reclaimed Surface Mines

Code and trained model for detecting autumn olive (*Elaeagnus umbellata*) from UAS imagery using random forest classification and U-Net semantic segmentation. Developed for my master's thesis at West Virginia University.

**Keane, S.M. (2025).** Machine Learning and UAS for Managing Autumn Olive (*Elaeagnus umbellata*) on Reclaimed Surface Mines. Master's Thesis, West Virginia University.

Funding: US Department of Interior, Office of Surface Mining Reclamation and Enforcement (Project No. S23AC00041-00)

Computational Resources: WVU Research Computing Dolly Sods HPC cluster (NSF OAC-2117575)

## Overview

Autumn olive is an invasive shrub that dominates reclaimed surface mines across the eastern United States. This project uses multispectral and RGB imagery from a Sentera 6X sensor mounted on a DJI Matrice 300 RTK to classify autumn olive across four phenological stages (early, peak, late, senescence) at two sites in West Virginia.

The pipeline uses two complementary approaches:

- **Random forest** — pixel-level classification from 5 spectral bands and 15 vegetation indices. Permutation importance and SHAP analysis identify the most discriminative indices for each phenological stage, which then inform the input selection for the U-Net models.
- **U-Net** — semantic segmentation of 256x256 image chips in three input configurations: RGB (3-band), multispectral (5-band), and vegetation index composite (8-band, using the top indices identified by the random forest analysis).

The best-performing model was the **peak-stage RGB U-Net** with 96.7% overall accuracy and an autumn olive F1-score of 0.927.

## Repository Structure

```
autumn-olive-detection/
├── README.md
├── Machine Learning and UASs for Managing Autumn Olive (Elaeagnus umbellata) on Reclaimed Surface Mines.pdf
├── 01_preprocessing/
│   ├── vegetation_indices.py          # raster vegetation index calculation
│   ├── tabular_vegetation_indices.R   # tabular vegetation index calculation
│   ├── image_chipper.py              # chip creation from orthomosaics
│   └── chip_normalization.py         # flight-level z-score normalization
├── 02_random_forest/
│   └── random_forest.R              # random forest classification
└── 03_segmentation/
    ├── unet_model.py                # shared u-net architecture
    ├── unet_training.py             # u-net training pipeline
    ├── unet_prediction.py           # tiled inference on full orthomosaics
    └── unet_rgb_peak_best.pth       # trained peak-stage RGB model weights
```

## Pipeline

1. **Radiometric correction** — Sentera 6X post-processing script (ILS + calibration panel)
2. **Orthomosaic generation** — Agisoft Metashape (3 cm/px multispectral, 1.5 cm/px RGB)
3. **Vegetation indices** — `vegetation_indices.py` (raster) or `tabular_vegetation_indices.R` (tabular for RF)
4. **Point sampling** — RF training data extracted via buffered GPS points (0.1m radius) in ArcGIS Pro 3.4
5. **Random forest classification** — `random_forest.R` performs balanced sampling, VIF assessment, grid search, and feature importance analysis
6. **Image chipping** — `image_chipper.py` creates 256x256 chips with 50% overlap, rasterizes annotation shapefiles into class masks
7. **Normalization** — `chip_normalization.py` applies flight-specific z-score normalization for RGB and VI-composite inputs. Multispectral data is radiometrically calibrated by the sensor.
8. **U-Net training** — `unet_training.py` trains segmentation models with weighted focal loss, OneCycleLR scheduling, and early stopping
9. **Inference** — `unet_prediction.py` applies trained models to full orthomosaics using tiled inference with overlap blending

## Using the Trained Model

```python
# in unet_prediction.py, set:
MODEL_PATH = "03_segmentation/unet_rgb_peak_best.pth"
MODEL_TYPE = "rgb"
INPUT_IMAGE = "path/to/orthomosaic.tif"
OUTPUT_DIR = "path/to/output"
```

Outputs a georeferenced class prediction raster, confidence map, and color visualization.

## Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Background | Unannotated pixels (ignored in training) |
| 1 | Barren | Exposed soil, rock, roads |
| 2 | LowVeg | Grass, herbaceous vegetation |
| 3 | AutumnOlive | *Elaeagnus umbellata* canopy |
| 4 | OtherTree | All other tree canopies |

## Vegetation Indices

NDVI, NDRE, GNDVI, BNDVI, LCI, GCI, RECI, SRI, GRNDVI, OSAVI, EVI2, ReGI, GRVI, CVI, GBVI — calculated from 5-band multispectral imagery (Blue, Green, Red, RedEdge, NIR).

## Requirements

**Python:** torch, rasterio, numpy, albumentations, scikit-learn, tqdm

**R:** randomForest, caret, fastshap, usdm, ggplot2, dplyr, doParallel

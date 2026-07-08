# PlateStateNet - Food Consumption Level Classification

---

## Project Motivation

In real-world restaurant environments, understanding the state of a plate (clean, finished, or full) can support automation and improve service efficiency.

However, real-world images are highly variable due to lighting, camera quality, noise, and perspective. Models trained on clean or synthetic data often fail when deployed in real conditions due to the **synthetic-to-real domain gap**.

This project investigates whether **synthetic data generation and camera-based augmentations** can improve robustness and generalization to real-world images.

---

## Problem Statement

**Input:** Plate image (clean or degraded)  
**Output:** Consumption state classification (Clean / Finished / Full)

**Goal:** Enable automated monitoring of table states in restaurant environments.

---

## Dataset

### Synthetic Data
- Generated using **FLUX.1-dev**
- ~2000 images (3 classes)
- ~300 prompts per class (high diversity)
- Fully reproducible with fixed seeds

Classes:
- Clean (empty pristine plate)
- Finished (used plate with residue)
- Full (plate with food)

### Real Data
- ~45 real-world CCTV images
- Used for evaluation (and minor tuning)

---

## Data Generation & Augmentation

### Prompt-Based Generation
- Attribute-based prompt construction
- Controlled randomness
- High variability in plate appearance

### Camera Simulation Augmentations
To bridge the synthetic → real gap:

- Low resolution (downscaling)
- Gaussian blur (defocus)
- Sensor noise
- Brightness & contrast variations
- Camera tilt & perspective shift
- Color cast (white balance changes)
- Vignette effects
- JPEG compression artifacts

These simulate the **image formation process of real cameras**.

---

## Visual Abstract

<p align="center">
  <img src="Images/data_examples.jpeg" width="900">
</p>

---

## Models and Pipeline

### Models
- ResNet18 (Linear Probe + Fine-Tuning)
- ResNet50 (Linear Probe + Fine-Tuning)

### Pipeline
1. Generate synthetic data  
2. Apply augmentations  
3. Train / validation / test split  
4. Train models (LP / FT)  
5. Evaluate on synthetic and real data  

---

## Training Process

- Models trained primarily on synthetic data  
- Fine-tuning allows adaptation to task-specific features  
- Small amount of real data used to slightly improve generalization  

---

## Metrics

- Accuracy  
- Confusion Matrix  

---

## Confusion Matrix Results

### ResNet18

<p align="center">
  <img src="Images/resnet18_results.png" width="900">
</p>

### ResNet50

<p align="center">
  <img src="Images/resnet50_results.png" width="900">
</p>

---

## Model Comparison (on test data)

| Model         | Synthetic Acc | Real Acc |
|---------------|---------------|----------|
| ResNet18 (LP) | 88.9%         | 91.1%    |
| ResNet18 (FT) | 95.6%         | 95.6%    |
| ResNet50 (LP) | 88.4%         | 91.1%    |
| ResNet50 (FT) | 94.7%         | 93.3%    |

---

## Dataset Access

The dataset is available via Google Drive:

👉 https://drive.google.com/drive/folders/1aZMwEyLMoWOu5gHtUaGOTgruwigfjvJQ

The Drive folder contains:

- Final processed datasets (after train/validation/test split)
- Real-world test samples

---

## Repository Structure

```
PlateStateNet/
│
├── code/            # training scripts & notebooks 
├── Images/          # figures used in README
├── visuals/         # results, plots, confusion matrices
├── Presentations/   # slides & reports
├── weights/         # trained model checkpoints
│
├── README.md
└── requirements.txt
```

---

## How to Run

This project follows a sequential pipeline. Each step is implemented as a notebook/script and should be executed in order.

Recommended: Run on a GPU machine (for FLUX and model training)

Install dependencies:
```
pip install -r requirements.txt
```

---

## Team Members

- Shlomi Ben Shitrit  
- Yarden Aviad  
- Avital Skop  

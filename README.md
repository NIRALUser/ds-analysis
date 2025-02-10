# DS Analysis Pipeline

This repository contains notebooks and scripts for analyzing brain imaging data through a multi-stage pipeline using self-organizing maps (SOM) and heatmap generation.

## Pipeline Overview

```
Input Images
     │
     ▼
[Neural Network]
     │
     ├─────→ score-norms (B, 20) ────→ SOM Analysis ────→ Prototype Identification 
     │                                                              │
     │                                                              ▼
     │                                                      Gather Behavior Scores  ────→ Correlate
     │                                                                                       ▲
     └─────→ score-images (B,20,H,W,D) ────→ Likelihood Model ────→ Heatmaps (1,H,W,D) ──────┘ 
```

## Pipeline Components

### 1. Image Processing and Feature Extraction
- Input: Raw brain imaging data
- Neural network processes images to generate:
  - Score norms: Batch x 20 dimensional feature vectors
  - Score images: Batch x 20 x Height x Width x Depth tensors

### 2. SOM Analysis Branch
- Takes score-norms as input
- Trains Self-Organizing Map for pattern discovery
- Identifies clusters (prototypes)
- Each sample will be matched to a protoype
- Enables visualization of data distribution

### 3. Heatmap Generation Branch  
- Takes score-images as input
- Feeds data through likelihood estimation model
- Generates 3D heatmaps (H x W x D) highlighting relevant regions
- Provides visualization of model attention/focus areas

### 4. Correlation Analysis
- Combines outputs from both branches
- Correlates behavioral scores with heatmap patterns
- Enables statistical analysis of relationships between:
  - Brain structural patterns
  - Behavioral metrics
  - Prototype membership

## Usage

### Requirements
- `braintypicality-scripts` repository
    - Contains all necessary code for preparing and processing the data
- `sade` package available at `https://github.com/ahsanMah/sade`
- `simpsom` package avilable at `https://github.com/ahsanMah/simpsom`
    ```bash
    cd /codespace/ && git clone https://github.com/ahsanMah/simpsom
    cd /codespace/simpsom && python setup.py install --user
    ```
- Additionally for plotting, `holoviews` and `hvplot` is used
- Other requirements (seaborn, pandas, etc.) should be covered by the packages above


### Running the Pipeline
1. Data Preparation
```bash
python prepare_data.py --input_dir /path/to/images --output_dir data/processed
```

2. Score Norm Feature Extraction

3. SOM Analysis
```python

```

4. Heatmap Generation
```python

```

## Data Formats

### Inputs
- Raw Images: NIfTI format (.nii.gz)
- Behavioral Scores: CSV files with subject IDs and metrics

### Intermediate Data
- score-norms: NumPy arrays (N x 20)
- score-images: NumPy arrays (N x 20 x H x W x D)

### Outputs
- SOM prototype assignments: CSV mapping samples to clusters
- Heatmaps: 3D NIfTI files (.nii.gz)
- Correlation matrices: CSV files

## Visualization

The pipeline includes visualization tools for:
- SOM topology and clustering
- Prototype patterns
- Heatmap overlays
- Correlation matrices

See the `notebooks/` directory for interactive visualization examples.

## References

[Include relevant papers and technical documentation]
# DS Analysis Pipeline

This repository contains notebooks and scripts for analyzing brain imaging data through a multi-stage pipeline using self-organizing maps (SOM) and heatmap generation.

## Pipeline Overview

```
Input Images
     │
     ▼
Preprocess + Register
     │
     ▼
[Neural Network]
     │
     ├─────⮞ score-norms (B, 20) ────⮞ SOM Analysis ────⮞ Prototype Identification 
     │                                                              │
     │                                                              ▼
     │                                                      Gather Behavior Scores  ────⮞ Correlate
     │                                                                                       ▲
     └─────⮞ score-images (B,20,H,W,D) ────⮞ Likelihood Model ────⮞ Heatmaps (1,H,W,D) ─────┘ 
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
  - Please install and run the docker container
- `simpsom` package avilable at `https://github.com/ahsanMah/simpsom`
    ```bash
    cd /codespace/ && git clone https://github.com/ahsanMah/simpsom
    cd /codespace/simpsom && python setup.py install --user
    ```
- Additionally for plotting, `holoviews` and `hvplot` is used
- Other requirements (seaborn, pandas, etc.) should be covered by the packages above


### Running the Pipeline
#### Data Preparation

We need to prepare data to be ingested by `sade`. Namely, the images should be two-channels (i.e. 4D images) with T1 and T2 concatenated. Please refer to the `sade` documentation for more details. You may use the `run_preprocessing.py` script to process the data.

To run inference sade will need a file e.g. `ibis-inlier.txt` with filenames of the images, and a directory where this file is present as specified by `--config.data.splits_dir`.

#### Score-Norm Extraction and Heatmap Generation

The inference sript of `sade` can extract score-norms and create heatmaps. The outputs will be saved as numpy files: `<workdir>/experiments/<experiment.id>/<subject-id>.npz`. An example config is given below.

```
python main.py --mode inference \
  --config configs/flows/gmm_flow_config.py \
  --workdir remote_workdir/cuda_opt/learnable/ \
  --config.data.splits_dir /ASD/ahsan_projects/Developer/braintypicality-scripts/split-keys \
  --config.eval.checkpoint_num=150 \
  --config.eval.experiment.flow_checkpoint_path=remote_workdir/cuda_opt/learnable/flow/psz3-globalpsz17-nb20-lr0.0003-bs32-np1024-kimg300_smin1e-2_smax0.8 \
  --config.eval.experiment.train=abcd-train  \
  --config.eval.experiment.inlier=abcd-val \
  --config.eval.experiment.id=default-ckpt-150 \
  --config.flow.patch_batch_size=16384 # Increasing this can help speed up inference - keep it powers of 2
```

> [!NOTE]
> These heatmaps are only *rigidly* registered to MNI. Any downstream voxel-wise analysis will need them to be deformably registered to the same space.

It is possible to use the `sade_registration.py` script in `braintypicality-scripts` to deformably register to MNI. If you choose to use `sade_registration.py`, here is an example:

```bash
# This will *compute* the registrations for the conte dataset
# and save them to the transforms directory specified in the code.
python sade_registration.py --mode compute \
--config /codespace/sade/sade/configs/ve/biggan_config.py \
--dataset conte

# This will *apply* the registrations from the transforms directory
python sade_registration.py --mode apply \
--config /codespace/sade/sade/configs/ve/biggan_config.py \
--dataset conte \
--load_dir /ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/experiments/reprod-correct/conte \
--save_dir /ASD/ahsan_projects/Developer/ds-analysis/ebds/registered-heatmaps/
```

#### SOM Analysis

The `build-som-plots` notebook can be run to produce the SOM clustering and the CSV of samples alongside their cluster IDs. This csv is used by the `roi_correlation_analysis` notebook

#### Heatmap Plotting

Heatmaps are stored as Numpy files. So they may be loaded and plotted using any preferred tools. An example is available in the `voxel-heatmaps` notebook. This will plot the average heatmap across the Down Syndrome samples that belong to the prototype. Recall, that computing the average only makes sense if the images are properly registered to the same space. 

### Reproducing the ds-analysis

The score-based diffusion model was trained with the following commands. This assumes the script is run from the `sade` folder inside the `sade` repository. All commands were run from inside the docker container produced by `sade/docker`.

```bash
python main.py --project architecture --mode train \
--config configs/ve/biggan_config.py
--workdir /workdir/cuda_opt/learnable \
--config.data.cache_rate=1.0 \
--config.model.learnable_embedding \
--config.training.batch_size=8 \ # switched to 16 at ~1 million iter
--cuda_opt
```

The flow model was run using
```bash
python main.py --project flows --mode flow-train \
--config configs/flows/gmm_flow_config.py \
--workdir /ASD/ahsan_projects/braintypicality/workdir/cuda_opt/learnable/ \
--config.data.cache_rate=1 \
--cuda_opt=0 \
--config.msma.min_timestep=1e-2 \
--config.msma.max_timestep=0.8 \
--config.flow.training_kimg=300
```
The flow model will be created in `<workdir>/flow/psz3-globalpsz17-nb20-lr0.0003-bs32-np1024-kimg300_smin1e-2_smax0.8/`. Then inference was run with 

```bash
python main.py --mode inference \
--config configs/flows/gmm_flow_config.py \
--workdir remote_workdir/cuda_opt/learnable/ \
--config.eval.checkpoint_num=150 \
--config.eval.experiment.flow_checkpoint_path=remote_workdir/cuda_opt/learnable/flow/psz3-globalpsz17-nb20-lr0.0003-bs32-np1024-kimg300_smin1e-2_smax0.8 \
--config.eval.experiment.id=default-ckpt-150 \
--config.eval.experiment.train=abcd-test  \ # can select different cohorts
--config.eval.experiment.inlier=ibis-inlier \
--config.eval.experiment.ood=ibis-ds-sa \
--config.flow.patch_batch_size=16384
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
- Heatmap overlays
- Correlation matrices

See the `notebooks/` directory for interactive visualization examples.

## References

[Include relevant papers?]
# Muscle Time Series Data Aggregation, Analysis, & Deformation Tracking

![openarm-multisensor tracking exemplar](https://people.eecs.berkeley.edu/~lhallock/publication/hallock2020biorob/featured.png)

This repo contains code used to 
- import, manipulate, and visualize muscle time series data, including ultrasound, surface electromyography (sEMG), acoustic myography (AMG), and output force data streams; and
- track muscle deformation (i.e., contour motion) using optical flow from time series ultrasound frames.

**If you use this code for academic purposes, please cite the following publication**: Laura A. Hallock, Akash Velu, Amanda Schwartz, and Ruzena Bajcsy, "[Muscle deformation correlates with output force during isometric contraction](https://people.eecs.berkeley.edu/~lhallock/publication/hallock2020biorob/)," in _IEEE RAS/EMBS International Conference on Biomedical Robotics & Biomechatronics (BioRob)_, IEEE, 2020.

**NOTE**: This (`master`) branch of this code contains the most recent stable version of our tracking and analysis code; it does not contain code for publications currently under review, but has been updated since the initial BioRob 2020 publication release. To access the codebase as released with the publication above, please visit the `biorob-2020` branch [here](https://github.com/lhallock/openarm-multisensor/tree/biorob-2020), and to access the version currently under review, visit `tnsre-dev` [here](https://github.com/lhallock/openarm-multisensor/tree/tnsre-dev).

This README primarily describes the methods needed to recreate the analyses described in the publications above, as applied to the time OpenArm Multisensor 1.0 data set found in the [OpenArm repository](https://simtk.org/frs/?group_id=1617). The code and documentation are provided as-is; however, we invite anyone who wishes to adapt and use it under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## Installation

### Downloading this repository

To download all modules and scripts, clone this repository via

```bash
git clone https://github.com/lhallock/openarm-multisensor.git
```

### Dependencies

All packages used in code development and their associated versions can be found in [`requirements.txt`](requirements.txt); however, many of these packages relate to our formatting, linting, and testing procedures and are unnecessary for non-developers. For simply running the code, the following Python modules are required, all of which can be installed via `pip`: `matplotlib`, `numpy`, `opencv-python`, `pandas`, `scipy`, and `seaborn`. **In particular, ensure that `opencv-python` and `pandas` are updated to the listed versions**; older installations may cause errors.

---

## Time series data aggregation, analysis, and plotting

This section describes the file structure and code necessary to recreate all plots and statistics in the publication above. Two main scripts are included: the first, [`run_multisensorimport_w_tracking.py`](run_multisensorimport_w_tracking.py) aggregates and plots all time series force, sEMG, and ultrasound-based deformation data, both ground-truth and for an indicated tracker, and writes out correlation tables to CSV; the second, [`gen_pub_figs.py`](gen_pub_figs.py), uses these CSV files and others generated in the deformation tracking procedures below to generate all bar plots and statistics reported in the publication above.

### Setup

Data should be downloaded from the `time_series` folder of the [OpenArm Multisensor 1.0 data set](https://simtk.org/frs/?group_id=1617) and arranged as follows:

```bash
.
├── gen_pub_figs.py
├── run_multisensorimport_w_tracking.py
├── sandbox/data/FINAL
│   ├── sub[N]
│   │   ├── seg_data.mat
│   │   ├── wp[i]t[j]
│   │   │   ├── ground_truth_csa.csv
│   │   │   ├── ground_truth_thickness.csv
│   │   │   ├── ground_truth_thickness_ratio.csv
│   │   │   ├── BFLK-G
│   │   │   │   ├── iou_series.csv
│   │   │   │   ├── tracking_csa.csv
│   │   │   │   ├── tracking_thickness.csv
│   │   │   │   └── tracking_thickness_ratio.csv
│   │   │   ├── BFLK-T
│   │   │   │   ├── iou_series.csv
│   │   │   │   ├── ...
│   │   │   ├── FRLK
│   │   │   │   ├── iou_series.csv
│   │   │   │   ├── ...
│   │   │   ├── LK
│   │   │   │   ├── iou_series.csv
│   │   │   │   ├── ...
│   │   │   ├── SBLK-G
│   │   │   │   ├── iou_series.csv
│   │   │   │   ├── ...
│   │   │   └── SBLK-T
│   │   │       ├── iou_series.csv
│   │   │       ├── ...
│   │   ├── ...
│   ├── ...

```

i.e., data should be placed in directory `sandbox/data/FINAL`, where `sandbox` is a directory at the top level of this repository. Alternatively, file paths can be modified via the constant variables at the top of each script.

Note that this file structure is consistent with the released ZIP archive; the high-level folder should simply be copied to the correct location. Some of the included files are raw/archive sensor data, while others are CSV tables generated by the deformation tracking code below. For a full discussion of all files and their origins, consult the README of the [data release](https://simtk.org/frs/?group_id=1617).

### Usage

First, run

```bash
python run_multisensorimport_w_tracking.py
```

to view all time series data plots. This will also generate correlation table files `ang_corr.csv` and `subj_corr.csv` in `sandbox/data/FINAL`, which are necessary for the second script below. (These files are also included with the OpenArm data release; the command above is thus optional if these files are already in the tree.)

Next, run

```bash
python gen_pub_figs.py
```

to view all bar plots and aggregate correlation statistics in the publication above.

Note that within both scripts, plots must be manually closed before the next will appear.

---

## Deformation tracking in ultrasound scans

This section describes the file structure and code necessary to recreate all muscle contour tracking results in the publication above (which can be readily adapted to track new image structures). Tracking (including both extraction of ground truth contour values from manually segmented data and contour tracking via optical flow) is accomplished via the included script [`run_tracking.py`](run_tracking.py), which performs both visualization and generation of CSV tracking error time series.

### Setup

Time series ultrasound data should be downloaded from the `ultrasound_frames` folder of the [OpenArm Multisensor 1.0 data set](https://simtk.org/frs/?group_id=1617), which includes both raw ultrasound image frames (`sub[N]/wp[i]/t[j]/raw`) and (for select trials) corresponding frames in which the brachioradialis contour has been manually segmented (`sub[N]/wp[i]t[j]/seg`). Because the code evaluates tracking quality against these ground truth scans, both must be downloaded to use the current release.

Paths to each folder are specified as command line arguments during script usage, so data may be stored anywhere. For instance,

```bash
.
├── run_tracking.py
├── sandbox/data/FINAL_FRAMES
│   ├── sub[N]
│   │   ├── wp[i]t[j]
│   │   │   ├── seg
│   │   │   │   ├── [frame_no].pgm
│   │   │   │   ├── ...
│   │   │   └── raw
│   │   │       ├── [frame_no].pgm
│   │   │       ├── ...
│   │   ├── ...
│   ├── ...
```

Note that within a single `sub[N]/wp[i]t[j]` folder, **the `seg` and `raw` folders should contain the same number of PGM files with the same frame numbers**. This ensures that the ground truth segmented frame is properly matched with its corresponding raw ultrasound frame when evaluating tracking quality.

### Usage

Run

```bash
python run_tracking.py --run_type <alg_int> --img_path <filepath_us> --seg_path <filepath_seg> --out_path <filepath_out> --init_img <filename_init>
```

specifying the above command line arguments as follows:

- `alg_int`: integer value corresponding to desired contour tracking algorithm
  - `1`: Naive Lucas&ndash;Kanade (LK)
  - `2`: Feature-Refined Lucas&ndash;Kanade (FRLK)
  - `3`: Bilaterally-Filtered Lucas&ndash;Kanade (BFLK)
  - `4`: Supporter-Based Lucas&ndash;Kanade (SBLK)
- `filepath_us`: file path to raw ultrasound PGM frames relative to run script (e.g., `sandbox/data/FINAL_FRAMES/sub[N]/wp[i]t[j]/raw/`)
- `filepath_seg`: file path to ground truth segmented PGM images relative to run script (e.g., `sandbox/data/FINAL_FRAMES/sub[N]/wp[i]t[j]/seg/`)
- `filepath_out`: file path to which `ground_truth_csa.csv`, `ground_truth_thickness.csv`, `ground_truth_thickness_ratio.csv`, `tracking_csa.csv`, `tracking_thickness.csv`, `tracking_thickness_ratio.csv`, and `iou_series.csv` time series tracking data will be written
- `filename_init`: file name of first image in ultrasound frame series (i.e., PGM file with the lowest frame number for a given trial; e.g., `618.pgm`)

If desired, parameter values associated with each tracking method can be modified via the static variables at the top of `run_tracking.py`.

### Parameter values

Each of the supported optical flow tracking methods can be tuned via a number of hyperparameters, which can be modified via the static variables at the top of `run_tracking.py`. A full list of hyperparameters, descriptions, and values used in the publications above can be found [here](params.md).

---

## Contributing

If you're interested in contributing or collaborating, please reach out to `lhallock [at] eecs.berkeley.edu`. If you're already a contributor to the project, please follow the code formatting and linting guidelines found [here](README_DEV.md).


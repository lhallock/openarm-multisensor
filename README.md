# Muscle Time Series Data Aggregation & Analysis

![openarm-multisensor tracking exemplar](header_animated.gif)

This repo contains code used to import, manipulate, and visualize muscle time
series data, including ultrasound, surface electromyography (sEMG), acoustic
myography, and output force data streams. Code used to generate the time series
data used in this repository (including muscle deformation tracking via optical
flow) can be found [here](https://github.com/lhallock/us-streaming) and
[here](https://github.com/cmitch/amg_emg_force_control)
(currently under development for open-source release).

**If you use this code for academic purposes, please cite the following publication**: Laura A. Hallock, Bhavna Sud, Chris Mitchell, Eric Hu, Fayyaz Ahamed, Akash Velu, Amanda Schwartz, and Ruzena Bajcsy, "[Toward Real-Time Muscle Force Inference and Device Control via Optical-Flow-Tracked Muscle Deformation](https://people.eecs.berkeley.edu/~lhallock/publication/hallock2021tnsre/)," in _IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)_, IEEE, 2021. (under review)

**NOTE**: This code branch has been updated in preparation for the paper submission above, currently under review. To access the codebase as released with the previous BioRob 2020 publication, please visit the `biorob-2020` branch [here](https://github.com/lhallock/openarm-multisensor/tree/biorob-2020), or view the last stable code release on the `master` branch [here](https://github.com/lhallock/openarm-multisensor/).

This README primarily describes the methods needed to recreate the analyses described in the publication above, as applied to the OpenArm Multisensor 2.0 data set found in the [OpenArm repository](https://simtk.org/frs/?group_id=1617). The code and documentation are provided as-is; however, we invite anyone who wishes to adapt and use it under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## Installation

### Downloading this repository

To download all modules and scripts, clone this repository via

```bash
git clone https://github.com/lhallock/openarm-multisensor.git
```

and navigate to this branch via

```bash
git checkout tnsre-dev
```

### Dependencies

All packages used in code development and their associated versions can be found in [`requirements.txt`](requirements.txt); however, many of these packages relate to our formatting, linting, and testing procedures and are unnecessary for non-developers. For simply running the code, the following Python modules are required, all of which can be installed via `pip`: `matplotlib`, `numpy`, `pandas`, `scipy`, and `seaborn`. **In particular, ensure that `pandas` is updated to the listed version**; older installations may cause errors.

---

## Time series data aggregation, analysis, and plotting

This section describes the file structure and code necessary to recreate all plots and statistics in the publication above. This is accomplished through the main script, [`gen_pub_figs.py`](gen_pub_figs.py), which generates all bar plots and statistics from the  [OpenArm Multisensor 2.0 data set](https://simtk.org/frs/?group_id=1617), as detailed below.

### Setup

Select data should be downloaded from the `time_series` and `survey` folders of the [OpenArm Multisensor 2.0 data set](https://simtk.org/frs/?group_id=1617) and arranged as follows:

```bash
.
├── gen_pub_figs.py
├── sandbox/data/FINAL
│   ├── [N]
│   │   ├── trial_0.p # optional, not used in analysis
│   │   ├── trial_1a.p # optional, not used in analysis
│   │   ├── trial_1b.p
│   │   ├── trial_2a.p # optional, not used in analysis
│   │   ├── trial_2b.p
│   │   ├── trial_3a.p # optional, not used in analysis
│   │   └── trial_3b.p
│   ├── ...
│   ├── survey_comp.csv
│   ├── survey_emg.csv
│   └── survey_emg.csv
│
```

i.e., data should be placed in directory `sandbox/data/FINAL`, where `sandbox` is a directory at the top level of this repository. Alternatively, file paths can be modified via the constant variables at the top of the script.

Note that this file structure is consistent with the released ZIP archive; all content in the high-level `time_series` folder, and the three survey response CSVs from `survey`, should simply be copied to the correct location. For a full discussion of all files and their origins, consult the README of the [data release](https://simtk.org/frs/?group_id=1617).

### Usage

Run

```bash
python gen_pub_figs.py
```

to view all bar plots and aggregate correlation statistics in the publication above.

Note that plots must be manually closed before the next will appear.

---

## Deformation tracking in ultrasound scans

The [OpenArm Multisensor 2.0 data set](https://simtk.org/frs/?group_id=1617)
used by this branch of the repository was generated using our [real-time muscle
thickness tracking system](https://github.com/lhallock/us-streaming), and thus this branch no longer contains code for muscle contour tracking. The old contour tracking code can still be accessed on the `biorob-2020` branch [here](https://github.com/lhallock/openarm-multisensor/tree/biorob-2020).

---

## Contributing

If you're interested in contributing or collaborating, please reach out to `lhallock [at] eecs.berkeley.edu`. If you're already a contributor to the project, please follow the code formatting and linting guidelines found [here](README_DEV.md).

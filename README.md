# Muscle Time Series Data Aggregation, Analysis, & Deformation Tracking

This repo contains code used to 
- import, manipulate, and visualize muscle time series data, including ultrasound, surface electromyography (sEMG), acoustic myography (AMG), and output force data streams; and
- track muscle deformation (i.e., contour motion) using optical flow from time series ultrasound frames.

**If you use this code for academic purposes, please cite the following publication**: Laura A. Hallock, Akash Velu, Amanda Schwartz, and Ruzena Bajcsy, "[Muscle deformation correlates with output force during isometric contraction](https://people.eecs.berkeley.edu/~lhallock/publication/hallock2020biorob/)," in _IEEE RAS/EMBS International Conference on Biomedical Robotics & Biomechatronics (BioRob)_, IEEE, 2020.

This README primarily describes the methods needed to recreate the analyses described in the publication above, as applied to the time series "multisensor" data found in the [OpenArm repository](https://simtk.org/frs/?group_id=1617). The code and documentation are provided as-is; however, we invite anyone who wishes to adapt and use it under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Installation

The following Python modules are required to run this code: `numpy`, `tensorflow`, `os`, `sys`, `math`, `logging`, `argparse`, `configparser`, `pickle`, `shutil`, `nibabel`, `scipy`, `gc`, `time`, `timeit`, and `collections`.

## Setup

Source data and models should be organized in the following directory structure:

```bash
├── models
│   ├── u-net_v1-0
│   │   ├── model_1
│   │   ├── model_2
│   │   └── folder_holding_multiple_models
│   │       ├── model_3
│   │       ├── model_4
│   │       └── ...
│	├── ...
├── Sub1
│   ├── all_nifti
│   ├── predictions
│   │   ├── over_512
│   │   │   ├── group_1_1
│   │   │   ├── group_1_2
│   │   │   └── ...
│   │   └── under_512
│   │       ├── group_1_1
│   │       ├── group_1_2
│   │       └── ...
│   ├── prediction_sources
│   │   ├── over_512
│   │   │   ├── trial11_60_fs
│   │   │   ├── trial12_60_gc
│   │   │   └── ...
│   │   ├── under_512
│   │   │   ├── trial1_0_fs
│   │   │   ├── trial10_30_p5
│   │   │   └── ...
├── Additional Sub[x] folders...
└── training_groups
    ├── group_1
    │   ├── group_1_1
    │   │   ├── trial11_60_fs
    │   │   └── ...
    │   ├── group_1_2
    │   │   ├── trial11_60_fs
    │   │   └── ...
    │   └── group_1_3
    │       ├── trial1_0_fs
    │       └── ...
    ├── group_2
    │   ├── group_2_1
    │   │   └── trial6_30_fs
    │   ├── group_2_2
    │   │   ├── trial6_30_fs
    │   │   └── trial7_30_gc
    │   └── group_2_3
    │       ├── trial10_30_p5
    │       ├── trial10_30_p5_ed
    │       ├── trial9_30_p3
    │       └── ...
    └── Additional training groups...

```

(The example structure above is reasonably comprehensive, aside from the `training_groups` folder, which is fleshed out more comprehensively below.)

Broadly, these directories contain the models used for training (`models`), 3D data and predictions for each target subject (`Sub1`, `Sub2`,...), and data used for training each specified network configuration (`training_groups`).

While some simpler training functionality can be executed without this structure — and some aspects of this structure are more fungible than others — it is necessary for scripts that automate training multiple models, generating segmentations of multiple images over multiple models, and scraping data from trained models or generated segmentations.

Each directory is described in further detail below.

### Models

TensorFlow model files are stored in the `models` directory. Each subdirectory (e.g., `u-net_v1-0`) contains all trained models of the same architecture. Inside each subdirectory are folders holding the actual Tensorflow model files, including both specified models and data that is logged during training. For example, `model_1` holds `checkpoint`, `model_1.data`, `model_1.index`, and `model_1.meta`, as well as the other information about the model that is stored during training (i.e., plain text and pickled files with loss per epoch, validation accuracies, etc.). Models can also be organized in a deeper folder structure if desired, provided the directory is properly specified in `trainingconfig.ini`, as described in the _Model Training_ section below.

Our own pre-trained models used in the publication above are available for download as part of the [OpenArm repository](https://simtk.org/frs/?group_id=1617). Other models may be used as well.

### Subjects

Subject folders `Sub[x]` contain all volumetric data associated with a given subject, including both raw volumetric scans and those that are populated by the network at prediction time. Subfolder `all_nifti` contains all raw NIfTI scans for which a predicted segmentation is desired (which should be formatted as `trial_*_volume.nii`). The `all_nifti` folder is used throughout the provided workflow as a comprehensive source of all available data (and metadata, such as NIfTI headers) for a given subject.

Each subject's `predictions` folder is populated at prediction time, though the directory itself (and its `over_512` and `under_512` subdirectories) must be created in advance. Specifically, when predictions are executed for a particular "group" (for which models and training data sources are specified in `trainingconfig.ini`), these prediction files are written into corresponding subfolders within each `predictions` folder.

Each subject's `prediction_sources` folder also contains all raw NIfTI scans for which prediction is desired, organized into the same `over_512` and `under_512` directories above, and then subfolders for each trial; this file structure should be created manually before prediction is attempted. Each subfolder must contain, at minimum, the trial's associated volume (`*_volume.nii`). If assessment of segmentation quality will be performed, as described in _Assessing Segmentation Quality_ below, each subfolder should contain the associated ground-truth segmentation as well (`*_seg.nii`). Note that both subfolders and NIfTI files should contain a `trial[n]_` prefix, and volume files should be named identically to their corresponding file in `all_nifti`. (The necessity of both these copies of each file is a redundancy that should be amended in future releases.) Volume filenames should include the characters `vol`, and segmentation filenames should include the characters `seg`.

Note that the `over_512` and `under_512` directories separate scans of which predicted slices are larger and smaller than 512x512 pixels. This is an artifact of the way the neural network generates predictions, which requires them to be padded to a power of two: scans smaller than 512x512 are padded to 512x512, while those larger are padded to 1024x1024. The full pipeline places them into separate folders, as they must be treated separately in the code's current instantiation. Note that this padding system works well for **generating predictions**, but padding larger scans to 1024x1024 results in significantly larger training times. To **train models** using larger scans, we recommend cropping to 512x512 instead.)

To predict segmentations for the available OpenArm 2.0 scans, first download all desired subject archives from the [project website](https://simtk.org/frs/?group_id=1617). All volume files for which predictions are desired (`Sub[x]/volumes/*_volume.mha`) should then be converted to the NIfTI file format (e.g., using [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php), renamed to follow convention `trial[n]_*_volume.nii`, and placed in the `all_nifti` folder, as well as corresponding subfolders in the `prediction_sources` subfolders. Available ground truth scans (`Sub[x]/ground_segs/*.nii` may also be placed in `prediction_sources` subfolders if available and prediction quality assessment is desired. Remember to place scans in the appropriate `over_512` and `under_512` directories according to their maximum dimension (aside from the dimension corresponding to the long axis of the arm, along which slices are collected). 

### Training Sources

The `training_groups` directory contains all data used to train each "group" (i.e. specified network and training data set for analysis). Below is an example subdirectory structure (for `training_groups/group_1` above; `group_2` is analogous):

```bash
group_1
├── group_1_1
│   ├── trial1_0_fs
│   │   ├── trial1_0_fs_seg.nii
│   │   └── trial1_0_fs_volume.nii
│   ├── trial11_60_fs
│   │   ├── trial11_60_fs_seg_cropped.nii
│   │   └── trial11_60_fs_volume_cropped.nii
│   ├── trial16_90_fs
│   │   ├── trial16_90_fs_seg_cropped.nii
│   │   └── trial16_90_fs_volume_cropped.nii
│   └── trial6_30_fs
│       ├── trial6_30_fs_seg.nii
│       └── trial6_30_fs_volume.nii
├── group_1_2
│   ├── trial1_0_fs
│   │   ├── trial1_0_fs_seg.nii
│   │   └── trial1_0_fs_volume.nii
│   ├── trial1_0_fs_ed
│   │   ├── trial1_0_fs_seg_ed.nii
│   │   └── trial1_0_fs_vol_ed.nii
│   ├── trial11_60_fs
│   │   ├── trial11_60_fs_seg_cropped.nii
│   │   └── trial11_60_fs_volume_cropped.nii
│   ├── trial11_60_fs_ed
│   │   ├── trial11_60_fs_seg_ed.nii
│   │   └── trial11_60_fs_vol_ed.nii
│   ├── trial16_90_fs
│   │   ├── trial16_90_fs_seg_cropped.nii
│   │   └── trial16_90_fs_volume_cropped.nii
│   ├── trial16_90_fs_ed
│   │   ├── trial16_90_fs_seg_ed.nii
│   │   └── trial16_90_fs_vol_ed.nii
│   ├── trial6_30_fs
│   │   ├── trial6_30_fs_seg.nii
│   │   └── trial6_30_fs_volume.nii
│   └── trial6_30_fs_ed
│       ├── trial6_30_fs_seg_ed.nii
│       └── trial6_30_fs_volume_ed.nii
```

Each `group_[j]_[k]` folder contains all data used for training that group, structured as a series of subfolders, each containing a single 3D volumetric scan and its associated segmentation. Note that, similarly to the requirements of `prediction_sources`, all subfolders and NIfTI filenames should contain the prefix `trial[n]_`, all volumes the characters `vol`, and all segmentations the characters `seg`.

## Model Training 

### Training a Single Model

To train a model, first ensure that the `models` and `training_groups` directories are structured as noted above, with the desired model and all desired training data.

Add an entry to `trainingconfig.ini` (or modify the `DEFAULT` entry), specifying (at minimum) the directories in which the desired model and training data are stored using the appropriate variables. You may safely use the hyperparameters specified in this repository's config file or may specify your own.

An additional important parameter may be set by modifying the source code of `training.py` directly; namely, the `total_keep` argument of each call to `split_data` specifies the total number of 2D image slices used from each scan (and, for the second call, augmented scan). Note that for general training purposes, these numbers should be set to 0, which is a special case that uses all available data; here, they are set relatively low to accommodate fair comparison of training methodologies for our associated publication.

To begin training, run

```bash
python training.py [model_name] -s [trainingconfig_section_name]
```

in terminal, where `[training_config_section_name]` corresponds to the section header of `trainingconfig.ini`, and `[model_name]` (which may be chosen as desired, conventionally as the same section header) specifies the directory inside `models` where training metadata will be stored.

### Queuing Training for Multiple Models

To train multiple models consecutively, follow all instructions above for training a single model, including directory setup and the addition of appropriate sections to `trainingconfig.ini`. Second, modify `trainmultiple.sh` to train the specific models desired. (Note that the example script here also contains examples of prediction, which can be eliminated if not necessary.) Training can then be accomplished via

```bash
sh trainmultiple.sh
```

## Predicting Segmentations

The easiest method of generating multiple predictions for multiple models is via the `predict_all_groups.py` script. Assuming the directory structure above (with special attention to the `Sub[x]` directories), modify the script variables `over_512_configs` and `under_512_configs` to contain the appropriate directory paths. Note that each variable contains a list of 3-tuples, each of which contains as first element the path to where the volumes to be used for prediction are stored, as second the path to where the predictions will be saved, and as third the path to all the available volumes for the given subject. As described above, these files are separated by size based on padding; for more information on why this is necessary, see the _Setup/Subjects_ section above.

Next, change the assignment of `models_dir` to a directory containing all models that will be used in prediction. (For example, in the file tree shown above, this could be the path to `u-net_v1-0` or to `folder_holding_multiple_models`. Note, however, that subfolders will not be searched recursively, so in the former case no models in the `folder_holding_multiple_models` subfolder will be used in prediction.

Lastly, the variable `group_whitelist` should be changed to include the names of all models (i.e., model directory subfolders) to be used in prediction. (For example, if `models_dir` is set to the `unet_v1-0` path, add `model_1` to include it or omit it to exclude it.) This whitelist structure is convenient for development, but can be easily discarded if desired by removing the `if group not in group_whitelist` check.

Once these variables are set, prediction can be accomplished via

```bash
python predict_all_groups.py
```

Note that if only a small number of models are in development, drawing on individual methods from the TensorFlow library and `src/pipeline.py` may be more straightforward. Models saved with the provided `training.py` script are saved using `tf.train.Saver` and can thus be restored with a call to the `tf.train.Saver.restore` method; this is the logic used within the provided `save_model` and `load_model` methods. Once a model is loaded, `predict_whole_seg` can be used to generate a prediction of a single NIfTI scan, and `predict_all_segs` to generate segmentations for all NIfTI files in a given directory.

## Assessing Segmentation Quality

The most straightforward way to assess the quality of many segmentations at once is to use the `generate_accuracy_table.py` script, which generates a table displaying prediction quality for each desired model with respect to a specified ground truth segmentation. By default, each table row corresponds to a trained network ("model group"), and each column displays the segmentation accuracy of that model group on a specific NIfTI scan. 

This table can be generated via

```bash
python generate_accuracy_table.py [accuracy_metric]
```

where `[accuracy_metric]` is either `mean_iou` (to calculate the mean intersection over union accuracy for biceps and humerus segmentations) or `total_percent` (to calculate the total percentage of correctly identified pixels from the entire scan). The script will save both a plaintext and pickled version of the table.

The following parameters should be edited to be consistent with your particular directory setup:

- `base_path` — path to the file tree described above (specifically, the path to all `Sub[x]` folders)

- `subjects` — list of all assessed subject identifiers (i.e., `[x]` for each `Sub[x]`)

- `size_dirs` — size-separated directories; should not require modification unless new scan sizes are desired

- `cols` — list of all column headers, formatted (after the first) as `trial[n][x]` for desired trial number `[n]` and subject identifier `[x]`

- `trial_mapping` — list mapping each column header in `cols` to its column number (should be in the same order as `cols`; used to index into the table and update each cell accordingly)

- `groups` — list of all networks / groups for which prediction assessment is desired (exactly the names of corresponding folders in the `predictions` directory of a given subject, assuming predictions were generated using the methods described in _Predicting Segmentations_ above)

Note that if your subject identifiers are numbers instead of letters, you may want to refactor this file for your own sanity so that numbers are not concatenated. The code should still function, however, as cells are filled by indexing into particular subject folders, then placing the value at the correct place in the table based on `trial_mapping`, rather than looping through the table and filling each cell in order based on `cols`, which could result in ambiguity.

## Training with Augmented Data

Our best performing networks in the above publication make use of augmented data, which can be generated from existing NIfTI files using the provided Jupyter Notebooks. Rotated and elastically deformed data can be generated using `rotate_nifti.ipynb` and `elastic_deform_nifti.ipynb`, respectively, and the new NIfTI scans generated can be used in training by placing them in the appropriate `training_groups` subdirectory as described above.

### Rotational Augmentation

Individual NIfTI scan pairs (volume + segmentation) can be rotated around the long axis of the arm using `rotate_nifti.ipynb`. Run

```bash
jupyter notebook rotate_nifti.ipynb
```

and modify `nii_data_dir`, `nii_vol_name`, and `nii_seg_name` to the appropriate path and filenames as indicated. Lastly, modify the `degrees` argument to specify the amount of rotation.

### Elastic Deformation

NIfTI files can be elastically deformed as individual volume + segmentation pairs or as multiple pairs from a single subject using `elastic_deform_nifti.ipynb`. Run

```bash
jupyter notebook elastic_deform_nifti.ipynb`
```

to access both of these functionalities.

To elastically deform a single NIfTI pair, modify `nii_data_dir`, `nii_vol_name`, and `nii_seg_name` to the appropriate path and filenames indicated in the top section. If desired, modify the `alpha` and `sigma` parameters, which specify the amount of displacement individual pixels experience and the amount of smoothing, respectively.

To elastically deform multiple NIfTI pairs from the same subject, place all NIfTI files for which augmentation is desired in a single directory. Additionally, ensure that the subject's `all_nifti` directory is set up in accordance with the file structure above, as it is used to extract header information for generating the new augmented NIfTI files. Modify `to_deform_dir`, `all_nifti_dir`, and `nii_save_dir` with appropriate file paths, and add all desired alpha and sigma values to `alphas` and `sigmas`, respectively, within the `elastic_transform_all` function. The code will then generate elastically deformed scans for all alpha-sigma combinations and all scans in `to_deform_dir`.

## Registration-Based Segmentation

In addition to the CNN-based segmentation code above, we provide the registration-based segmentation code, built using [SimpleElastix](https://simpleelastix.github.io/), used as a baseline in the publication above. Its use is documented below.

Note that registration-based segmentation consists of mapping the segmented tissue structures of one scan to another by finding the optimal transformation between the the two raw images. An excellent description of this process can be found [here](https://simpleelastix.readthedocs.io/Introduction.html).

### Installation

Use of registration code relies on the `numpy` Python module and the [SimpleElastix](https://simpleelastix.github.io/) library.

### Setup

Edit the following parameters, at the top of the `run_amsaf` method in `registration.py`:

- `verbose` — set as `True` or `False` based on desired verbosity

- `unsegmented_image` — set as `read_image("[image_to_be_segmented].nii")` based on the file path of the NIfTI to be segmented

- `segmented_image`, `segmentation` — set as `read_image("[segmented_image].nii")` based on the file paths of the segmented NIfTI source image and its associated segmentation, respectively

- `new_segmentation` — set as desired output file name for new segmentation

If your segmented and unsegmented images are not already roughly aligned, you may choose to specify a manual affine transformation with which to initialize the registration process by modifying the `A` and `t` parameters.

By default, the provided code will perform a hierarchy of rigid, affine, and nonlinear transformations, with the result of each registration initializing the next. If you wish to more precisely control the behavior of these transformations, you may edit the `DEFAULT_*` parameter maps included at the bottom of `registration.py`.

### Usage

Run

```bash
python registration.py
```

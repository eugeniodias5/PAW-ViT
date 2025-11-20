
  

# Masks generation

  

We publish the code to generate the semantic segmentation pseudo-masks from segmentation masks and animal keypoint estimations. This repository is a fork of [mmpose](https://github.com/open-mmlab/mmpose).

  

## Table of Contents


- [Overview](#overview)

- [Requirements](#requirements)

- [Setup](#setup)  

- [Running](#training)

- [Visualization](#visualization)

## Overview
## Overview

To generate the masks, we follow four steps:

 1. We apply foreground segmentation using [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0).
 2. We extract keypoints using a [HRNet-W48](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html#ap10k-dataset) trained on the [Animal-Pose dataset](https://github.com/noahcao/animal-pose-dataset).  This model estimates 20 keypoints, but outputs few points on the animal's body, so we complement it with [SuperAnimal-Quadruped](https://www.nature.com/articles/s41467-024-48792-2). Details on how we map the keypoints can be find inside the folder `configs/kpts_mapping`.
 3. We interpolate synthetic keypoints along straight lines between adjacent visible keypoints, particularly on the animal's body. If all the keypoints are visible, we can insert up to 67 synthetic keypoints. Details on the interpolated keypoints can be find inside the folder `configs/kpts_mapping`. For the tigers, to deal with their curved tails, we calculate the Dijkstra shortest path between the keypoints "tail base" and "tail end" (extracted from SuperAnimal-Quadruped), and insert two additional synthetic keypoints along that path.
 4. Each keypoint is associated with a specific anatomical region. To generate the pseudo-semantic masks, we assign every pixel within the foreground mask to the region of its nearest keypoint. Details on the keypoints association to anatomical regions can be find inside the folder `configs/kpts_mapping`.
  
Below, you may find an example of keypoints estimation (on the left) and of semantic segmentation (on the right).

![Example of keypoints estimation and semantic segmentation pseudo-mask generation on a tiger of the ATRW dataset](resources/gt_masks_atrw_sup.png)


## Requirements
We used python 3.10.4.

## Setup

**1. Download the weights**
First, you need to download the following model weights:

 - **Detection:** [YOLOv11s](https://docs.ultralytics.com/fr/models/) - the model is automatically downloaded.
 - **Segmentation:** [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0).
 -  **Keypoint estimation:** [HRNet_w48](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html#ap10k-dataset) - Download HRNet_w48 trained on AP10k.
 -  **Keypoint estimation:** [hrnet_w32_quadruped80k.pth](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/tree/main) - Download the model hrnet_w32_quadruped80k.pth.

Place the models in your prefered folder and set the paths in the configuration file: `configs/config.yaml`.

2.  **Set up your environment**
We recommend you to use a different environment as the one used for PAW-ViT to avoid dependency conflicts.

Create your Python environment using `venv`:

`python -m venv env_pawvit`

 or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html):

`conda create --name env_test python=3.10.4 -y`

 **3. Install the dependencies**

Download the libraries using the command `pip install -r requirements.txt`. 

If you prefer, you can follow the [MMPose installation steps](https://mmpose.readthedocs.io/en/latest/installation.html).

***Attention:***
If You find the error:
AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.
Simply modify the variable 'mmcv_maximum_version' in mmdet/\_\_init__.py to a higher version, for example: mmcv_maximum_version = '2.2.1' :).

**4.  Customize the configuration file**
Customize the configuration file: `configs/config.yaml` adding the paths to models weights, datasets, and your preferences for keypoint threshold. The file is written as used in the paper.

## Running
To generate the masks, run the command:

    python main.py --config=[CONFIG PATH] --batch-size 1

Note that we only process the detection and the segmentation in batches because the MMPose native function [inference_topdown](https://mmpose.readthedocs.io/en/latest/api.html#mmpose.apis.inference_topdown) only supports one image at a time.

## Visualization
Additionally, you can use the methods `vis_keypoints` and `vis_masks` to visualize, respectively, the keypoints estimation and the semantic segmentation pseudo-masks. These methods were used to generate the masks shown in the paper.

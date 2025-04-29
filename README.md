# MDR-Net: Multi-Directional and Rotation-aware Network for Rotated Object Detection

Authors:[Quan Cui](https://github.com/cowqer)\*, [Gaodian Zhou](https://github.com/tist0bsc),[Xiaolin Zhu]

## Introduction

This is the official implementation of the paper, MDR-Net: Multi-Directional and Rotation-aware Network for Rotated Object Detection.
In this paper, we propose a two-stage framework called Multi-Directional and Rotationaware Network(MDR-Net), which consists of three key modules. (1) Gated Pinwheel-shaped
Convolution (GPC). The GPC enhances the detection of elongated targets aligned along horizontal and vertical axes by adaptively fusing receptive fields in orthogonal directions. (2) Rotated
Convolution module with Attention-guided routing (RCA). RCA constructs a Multi-Scale Convolutional Attention(MSCA) framework to capture rotation angles and weights, then uses rotational convolution kernels to extract the features, to reduce the feature differences in ships caused by varying orientations. (3)Feature-Aligned Oriented Region Proposal Network (FAORPN). To
generate proposals that more accurately localize multi-oriented and elongated targets, FAORPN is designed by integrating RCA and GPC through weighted fusion within the ORPN.

## The Gated Pinwheel-shaped Convolution

![GPC](https://github.com/user-attachments/assets/2feb6f28-58c2-4b99-b16b-6f113cb563f8)

## The GPC-R50

![GPC-R50](https://github.com/user-attachments/assets/b04aaf33-e895-46de-be46-2e160bf0f219)

## The Rotated Convolution module with Attention-guided routing 

![RCA](https://github.com/user-attachments/assets/a4dfb8b7-bb45-4ad1-8aea-90e9bdbedd52)


## The Achitecture of MDR-Net

![Architecture](https://github.com/user-attachments/assets/cfe143b9-c074-48bf-b315-9cc8a46e4b13)


## Installation

We ued the MMRotate toolbo, which depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

### DATA 

 DOTA: [official website](https://captain-whu.github.io/DOTA/dataset.html)  
 RSSDD:[Official-SSDD-OPEN.rar](https://pan.baidu.com/s/1HrlI6KM2dX7YrIBSZ7Hiuw?pwd=ssdd)

In the file ./configs/MDR-Net/_base_/datasets/dota.py or ssdd.py, change the data path following ```data_root``` to ```YOUR_DATA_PATH```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)
- 
## Usage

### Training

```
python tools/train.py  configs/MDR-Net/oriented_rcnn_gatedpc_r50_fpn_1x_dota_le90_msca_adp_rpn.py
```

### Test and Submit

```
python ./tools/test0.py \
configs/MDR-Net/oriented_rcnn_gatedpc_r50_fpn_1x_dota_le90_msca_adp_rpn.py \
YOUR_CHECKPOINT_PATH --eval mAP

python ./tools/test0.py \
configs/oriented_rcnn/oriented_rcnn_gatedpc_r50_fpn_1x_dota_le90_msca_adp_rpn.py \
YOUR_CHECKPOINT_PATH --gpu-ids 0 \
--format-only --eval-options \
submission_dir=YOUR_SAVE_DIR
```

## Acknowledgement

This code is developed on the top of [MMrotate](https://github.com/open-mmlab/mmrotate/), we thank to their efficient and neat codebase.

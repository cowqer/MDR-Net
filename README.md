# MDR-Net:

Authors: [Quan Cui](https://github.com/cowqer), Yan Zhou\*, [Gaodian Zhou\*](https://github.com/tist0bsc), Jianxun Li, Xiaolin Zhu, Richard Irampaye. [Multi-Directional Rotation-Aware Network for Oriented Ship Detection From Remote Sensing Imagery](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11226875)(IEEE JSTAR 2025).

## Introduction

This is the official implementation of the paper, MDR-Net: Multi-Directional Rotation-Aware Network for Oriented Ship Detection From Remote Sensing Imagery
In this paper, we propose a two-stage framework called Multi-Directional and Rotation aware Network(MDR-Net), which consists of three key modules. (1) Gated Pinwheel-shaped
Convolution (GPC). The GPC enhances the detection of elongated targets aligned along horizontal and vertical axes by adaptively fusing receptive fields in orthogonal directions. (2) Rotated
Convolution module with Attention-guided routing (RCA). RCA constructs a Multi-Scale Convolutional Attention(MSCA) framework to capture rotation angles and weights, then uses rotational convolution kernels to extract the features, to reduce the feature differences in ships caused by varying orientations. (3)Feature-Aligned Oriented Region Proposal Network (FAORPN). To
generate proposals that more accurately localize multi-oriented and elongated targets, FAORPN is designed by integrating RCA and GPC through weighted fusion within the ORPN.

## The Gated Pinwheel-shaped Convolution

<img src="https://github.com/user-attachments/assets/c76a4d1e-5b72-4cc5-b71e-36668f65266f" width="80%" />

## The GPC-R50

<img src="https://github.com/user-attachments/assets/72db284c-c0e0-4a37-9586-d98e22e415a5" width="80%" />

## The Rotated Convolution module with Attention-guided routing 

<img src="https://github.com/user-attachments/assets/3c3f448f-9569-49b5-a135-29bf920332c6" width="80%" />

## The Achitecture of MDR-Net

<img src="https://github.com/user-attachments/assets/6326815e-5f13-44e5-8e5d-6952815c5e02" width="80%" />

## Results and models

DOTA1.0
| Model         | mAP50 | mAP75 | Batch Size | Config | Download |
|---------------|--------|--------|------------|--------|----------|
| ORCNN         | 75.37 | 46.05 | 1×2        | [config](https://github.com/cowqer/MDR-Net/blob/main/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py) | [model (pswd: mdrn)](https://pan.baidu.com/s/1hmVfLerupdak8NbquBw4PA) |
| MDR-Net       | 75.89 | 48.24 | 1×2        | [config](https://github.com/cowqer/MDR-Net/blob/main/configs/MDR-Net/oriented_rcnn_gatedpc_r50_fpn_1x_dota_le90_msca_adp_rpn.py) | [model (pswd: mdrn)](https://pan.baidu.com/s/1hmVfLerupdak8NbquBw4PA) |
| MDR-Net (ms)  | 80.58 | 56.61 | 1×2        | [config](https://github.com/cowqer/MDR-Net/blob/main/configs/MDR-Net/oriented_rcnn_gatedpc_r50_fpn_1x_msdota_le90_msca_adp_rpn.py) | [model (pswd: mdrn)](https://pan.baidu.com/s/1fGsQj8Zf6JYWryWzQC3__A) |
RSSDD 
| Model    | mAP50  | mAP75 | Batch Size | Config | Download |
|----------|--------|--------|------------|--------|----------|
| MDR-Net  | 0.8935 | 41.5   | 1×2        | [config](https://github.com/cowqer/MDR-Net/blob/main/configs/MDR-Net/oriented_rcnn_gpcr50_fpn_6x_ssdd_le90_msca1.py) | [model (pswd: mdrn)](https://pan.baidu.com/s/10ZpDPSaYnr2bGbBieOrNsQ) |


## Installation
We ued the MMRotate toolbox, which depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
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
 RSSDD: [Official-SSDD-OPEN.rar](https://pan.baidu.com/s/1HrlI6KM2dX7YrIBSZ7Hiuw?pwd=ssdd)

In the file ./configs/MDR-Net/_base_/datasets/dota.py or ssdd.py, change the data path following ```data_root``` to ```YOUR_DATA_PATH```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

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

## Cite

```
@ARTICLE{11226875,
  author={Cui, Quan and Zhou, Yan and Zhou, Gaodian and Li, Jianxun and Zhu, Xiaolin and Irampaye, Richard},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Multidirectional Rotation-Aware Network for Oriented Ship Detection From Remote Sensing Imagery}, 
  year={2026},
  volume={19},
  number={},
  pages={190-208},
  keywords={Marine vehicles;Convolution;Feature extraction;Remote sensing;Kernel;Shape;Accuracy;Proposals;Object detection;Attention mechanisms;Oriented object detection;pinwheel-shaped convolution;remote sensing;rotational convolution kernel;ship detection},
  doi={10.1109/JSTARS.2025.3629101}}
```

## Acknowledgement

This code is developed on the top of [MMrotate](https://github.com/open-mmlab/mmrotate/), we thank to their efficient and neat codebase.

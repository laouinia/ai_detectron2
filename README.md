# Detectron2

## Installation & Setup

### Install

- Clone detectron2 and build it.

### Setup

- Build it.

## Device Hander

- Device Handler Cuda vs CPU

## Detection Types

- Tested the detectron2 with different types of detections:
  - Object detection
  - Keypoints Detection
  - Instant Segmentation
  - Instant Segmentation with Rend Point
  - Panoptic Segmentation

## Custom dataset

[Data Set](https://github.com/ari-dasci)
[Use Custom Datasets](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)

- Used a subset of OD-WeaponDetection: Knife_detection to build the custom dataset to train the model
- Used label_me to draw the boxes around the object
- Used label-me-2-coco to convert the json file into coco data format

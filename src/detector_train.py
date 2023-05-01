"""Detectron Train """
import os
import pickle
from typing import (
    Literal
)

import torch.cuda

from detector_utils import (
    cfg_train_dataloader,
)

from detectron2.config import (
    # get_cfg,
    CfgNode,
)

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer

setup_logger()

# Device handler
device: Literal['cuda', 'cpu'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# CONFIG_FILE_PATH = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
# CHECKPOINT_URL = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
# OUTPUT_DIR= '../output/object_detection'

# Saving the configuration into a pickle file
# CFG_SAVE_PATH = '../config/object_detection_config.pkl'

# Training for Instance Segmentation
CONFIG_FILE_PATH = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
CHECKPOINT_URL = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
OUTPUT_DIR= '../output/instance_segmentation'

# Saving the configuration into a pickle file
CFG_SAVE_PATH = '../config/instance_segmentation_config.pkl'

NUM_CLASSES: int = 1


# Train Dataset
TRAIN_DATASET_NAME = 'train_knives'
TRAIN_DATASET_PATH = '../data/train'
TRAIN_DATASET_JSON = '../data/train.json'

# Test Dataset
TEST_DATASET_NAME = 'test_knives'
TEST_DATASET_PATH = '../data/test'
TEST_DATASET_JSON = '../data/test.json'

#  Register Train Dataset
register_coco_instances(
    name=TRAIN_DATASET_NAME,
    metadata={},
    json_file=TRAIN_DATASET_JSON,
    image_root=TRAIN_DATASET_PATH
)

# Register Test Dataset
register_coco_instances(
    name=TEST_DATASET_NAME,
    metadata={},
    json_file=TEST_DATASET_JSON,
    image_root=TEST_DATASET_PATH
)


def train_object_detection() -> None:
    """Training Function for object detection"""
    # Loading the CFG
    cfg: CfgNode = cfg_train_dataloader(
         config_file_path=CONFIG_FILE_PATH,
         checkpoint_url=CHECKPOINT_URL,
         train_dataset_name=TRAIN_DATASET_NAME,
         test_dataset_name=TEST_DATASET_NAME,
         num_classes=NUM_CLASSES,
         device=device,
         output_dir=OUTPUT_DIR,
    )

    # save the cfg to use in the test file
    with open(CFG_SAVE_PATH, 'wb') as config_file:
        pickle.dump(cfg, config_file, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    train_object_detection()

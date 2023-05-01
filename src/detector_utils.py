"""Utils file for helper function."""
import warnings

from detectron2 import model_zoo
from detectron2.config import (
    get_cfg,
    CfgNode,
)

# Suppress UserWarning: torch.meshgrid
warnings.filterwarnings(
    "ignore",
    message=(
        'torch.meshgrid: in an upcoming release,'
        ' it will be required to pass the indexing argument.'
    )
)

def cfg_train_dataloader(
    config_file_path: str,
    checkpoint_url: str,
    train_dataset_name: str,
    test_dataset_name: str,
    num_classes: int,
    device: str,
    output_dir: str
) -> CfgNode:
    """Temp"""
    cfg: CfgNode = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2 # number of workers for dataloader (default: 2)
    cfg.SOLVER.IMS_PER_BATCH = 2 # number of images per batch
    cfg.SOLVER.BASE_LR = 0.00025 # initial learning rate
    cfg.SOLVER.MAX_ITER = 1000 # maximum number of iterations to run
    cfg.SOLVER.STEPS = [] # iterations to decay learning rate by 10x
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes # 1: knife is the default
    cfg.DEVICE = device # cuda or cpu
    cfg.OUTPUT_DIR = output_dir

    return cfg

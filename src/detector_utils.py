"""Utils file for helper function."""
import warnings
from numpy import (
    dtype,
    generic,
    ndarray,
)
import cv2
import matplotlib.pyplot as plt

from detectron2 import model_zoo
# from detectron2.data import (
#     DatasetCatalog,
#     MetadataCatalog,

# )

from detectron2.config import (
    get_cfg,
    CfgNode,
)

from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    # VisImage,
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

def detection_in_image(image_path: str, predictor) -> None:
    """Temp."""
    image: ndarray[int, dtype[generic]] = cv2.imread(image_path)
    outputs = predictor(image)
    vis: Visualizer = Visualizer(image[:, :, ::-1],
                     metadata={},
                    #  scale=0.3,
                     instance_mode=ColorMode.SEGMENTATION)
    vis = vis.draw_instance_predictions(outputs['instances'].to('cpu'))

    plt.figure(figsize=(15, 10))
    plt.imshow(vis.get_image())
    plt.show()


def detection_in_video(video_path: str, predictor) -> None:
    """Temp."""
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened() is False:
        return

    while True:
        video_is_on, frame = capture.read()
        if not video_is_on:
            break

        predictions = predictor(frame)
        vis = Visualizer(frame[:, :, ::-1],
                         metadata={},
                         instance_mode=ColorMode.SEGMENTATION)
        output = vis.draw_instance_predictions(predictions['instances'].to('cpu'))

        cv2.imshow('Detection in Video', output.get_image()[:, :, ::-1])

        key_press: int = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

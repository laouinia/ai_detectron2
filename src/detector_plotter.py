"""
Detector Plotter Module.

This module contains the function to plot images from a dataset.
It is used to check the dataset registration with coco.
"""
import random
from typing import Any
from numpy import (
    dtype,
    generic,
    ndarray,
)
from matplotlib import pyplot as plt

import cv2

from detector_train import TRAIN_DATASET_NAME
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    Metadata,

)
from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    # VisImage,
)

def plotter(dataset_name:str, num_samples: int = 1, seed: int = 1) -> None:
    """Plot images from a dataset.

    Args:
        dataset_name (str): name of the dataset.
        num_samples (int, optional): number of samples to plot. Defaults to 1.
        seed (int, optional): random seed. Defaults to 1.
    """
    random.seed(seed)
    dataset_dicts: Any = DatasetCatalog.get(dataset_name)
    metadata: Metadata | Any = MetadataCatalog.get(dataset_name)

    for simple_img in random.sample(dataset_dicts, num_samples):
        img: ndarray[int, dtype[generic]] = cv2.imread(simple_img['file_name'])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=metadata,
                                scale=0.5,
                                instance_mode=ColorMode.IMAGE_BW)
        vis = visualizer.draw_dataset_dict(simple_img)
        plt.figure(figsize=(15, 15))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()

if __name__ == '__main__':
    plotter(dataset_name=TRAIN_DATASET_NAME, num_samples=2)

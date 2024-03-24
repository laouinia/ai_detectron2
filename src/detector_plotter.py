"""
Detector Plotter Module.

This module contains the function to plot images from a dataset.
It is used to check the dataset registration with coco.
"""

import random

import cv2
from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer  # VisImage,
from matplotlib import pyplot as plt
from numpy import dtype, generic, ndarray

from detector_train import TRAIN_DATASET_NAME


def plotter(dataset_name: str, num_samples: int = 1, seed: int = 1) -> None:
    """Plot images from a dataset.

    Args:
        dataset_name (str): name of the dataset.
        num_samples (int, optional): number of samples to plot. Defaults to 1.
        seed (int, optional): random seed. Defaults to 1.
    """
    random.seed(seed)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata: Metadata = MetadataCatalog.get(dataset_name)

    for simple_img in random.sample(dataset_dicts, num_samples):
        img: ndarray[int, dtype[generic]] = cv2.imread(simple_img["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,
        )
        vis = visualizer.draw_dataset_dict(simple_img)
        plt.figure(figsize=(15, 15))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def main() -> None:
    """ main Entry Point """
    plotter(dataset_name=TRAIN_DATASET_NAME, num_samples=2)


if __name__ == "__main__":
    main()

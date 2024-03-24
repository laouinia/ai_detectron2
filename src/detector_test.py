"""Detectron Custom Testing Module."""

import os
import pickle
import tkinter as tk
from tkinter import filedialog

from detectron2.config import CfgNode
from detectron2.engine import DefaultPredictor

from detector_utils import detection_in_image, detection_in_video

# CFG_SAVE_PATH = '../config/object_detection_config.pkl'
CFG_SAVE_PATH = "../config/instance_segmentation_config.pkl"

# To load the configuration file
with open(CFG_SAVE_PATH, "rb") as config_file:
    cfg: CfgNode = pickle.load(config_file)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confident

predictor = DefaultPredictor(cfg)


def main() -> None:
    """
    Testing the model on a single image or a video.
    """
    object_input: str = input(
        "\nChoose Your Object Type:\n\n input i: images or v: videos?"
        "\n\n\tObject Type: "
    )

    root = tk.Tk()
    root.withdraw()  # hide the root window

    file_path: str = filedialog.askopenfilename()

    if object_input.lower() == "i":
        detection_in_image(file_path, predictor)
    elif object_input.lower() == "v":
        detection_in_video(file_path, predictor)
    else:
        print("Invalid Choice!")


if __name__ == "__main__":
    main()

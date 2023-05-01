"""Detectron2
        Remarques:
            ! OpenCV uses BGR: [Bleu-Green-Red]
"""
# pylint: disable=line-too-long
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
)
from numpy import (
    ndarray,
    dtype,
    generic
)

import torch.cuda

import cv2

from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import (
    get_cfg,
    CfgNode,
)
from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    VisImage
)
from detectron2.projects import point_rend

warnings.filterwarnings(
    "ignore",
    message=(
        'torch.meshgrid: in an upcoming release,'
        ' it will be required to pass the indexing argument.'
    )
)

# Device handler
device: Literal['cuda', 'cpu'] = 'cuda' if torch.cuda.is_available() else 'cpu'

class Detector:
    """ Class Detector to chose the detector type."""

    def __init__(self, model_type: str, object_type: str) -> None:
        self.cfg: CfgNode = get_cfg()
        self.model_type: str = model_type
        self.object_type: str = object_type

        model_type_int:int = int(model_type)
        model_type_dict: dict[int, Callable[[], None]] = {
            1: self._setup_object_detection,
            2: self._setup_keypoints_detection,
            3: self._setup_instant_segmentation,
            4: self._setup_instant_segmentation_point_rend,
            5: self._setup_panoptic_segmentation,
        }

        object_type_dict  = {
            'i': self.detect_object_in_images,
            'v': self.detect_object_in_video,
        }

        try:
            model_type_dict[model_type_int]()
        except (ValueError, KeyError):
            print(f'\nDetection Model Type: {model_type} is incorrect.')
            sys.exit()

        try:
            object_type_dict[object_type]
        except KeyError:
            print(f'\nObject model type: "{object_type}" is incorrect.')
            sys.exit()


        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = device
        self.predictor: DefaultPredictor  = DefaultPredictor(self.cfg)

    def _setup_object_detection(self) -> None:
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
            'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')

    def _setup_keypoints_detection(self) -> None:
        self.cfg.merge_from_file(model_zoo.get_config_file(
            'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')

    def _setup_instant_segmentation(self) -> None:
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

    def _setup_panoptic_segmentation(self) -> None:
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
            'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml')

    def _setup_instant_segmentation_point_rend(self) -> None:
        point_rend.add_pointrend_config(self.cfg)
        self.cfg.merge_from_file(
            '../detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml')
        self.cfg.MODEL.WEIGHTS = (
            'detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'
        )

    def detect_object_in_images(self, image_path: str) -> None:
        """
        Object detection in Images

        Args:
            image_path (str): Path to image input
        """
        # image = cv2.imread(image_path)
        image: ndarray[int, dtype[generic]]= cv2.imread(image_path)
        image_height: int
        image_width: int
        _: int
        image_height, image_width , _ = image.shape

        print(
            f'\nImage Properties:\n'
            f'\tHeight:\t {image_height}px\n'
            f'\tWidth:\t {image_width}px\n'
        )

        if self.model_type != 4:
            predictions: Dict[str, Any] = self.predictor(image)

            image_visualizer = Visualizer(
                image[:, :, : : -1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.IMAGE
            )
            output: VisImage = image_visualizer.draw_instance_predictions(predictions['instances'].to('cpu'))

        else:
            predictions, segment_info = self.predictor(image)['panoptic_seg']
            image_visualizer = Visualizer(
                image[:, :, : : -1],
                MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            )
            output = image_visualizer.draw_panoptic_seg_predictions(
                predictions.to('cpu'), segment_info
            )

        cv2.imshow('Result', output.get_image()[:, :, : : -1])

        key_press: int = cv2.waitKey(0)

        while key_press and key_press != ord('q'):
            pass

    def detect_object_in_video(self, video_path: str) -> None:
        """
        Object detection in a video

        Args:
            video_path (str): Path to input video
        """
        capture: Any = cv2.VideoCapture(video_path)

        if capture.isOpened() is True:
            print(
                f'\nVideo Properties:\n'
                f'\tWidth:\t {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}xp\n'
                f'\tHeight:\t {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}xp\n'
                f'\tFPS:\t {capture.get(cv2.CAP_PROP_FPS)}fps'
            )

        else:
            print('\n\t!!! File cannot be open or found.')
            sys.exit(0)

        # Type fix : bool, ndarray
        video_is_on, image = capture.read()

        while video_is_on:
            if self.model_type != '4':
                predictions: Dict[str, Any] = self.predictor(image)

                visualizer: Visualizer = Visualizer(
                    image[:, :, : : -1],
                    metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                    instance_mode=ColorMode.IMAGE
                    )
                output: VisImage = visualizer.draw_instance_predictions(predictions['instances'].to(device='cpu'))

            else:
                predictions, segment_info = self.predictor(image)['panoptic_seg']
                visualizer = Visualizer(
                    image[:, :, : : -1],
                    MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                )
                output = visualizer.draw_panoptic_seg_predictions(
                    predictions.to('cpu'), segment_info
                )

            # else:
            #     break

            cv2.imshow('Detection in Video', output.get_image()[:, :, : : -1])
            video_is_on, image = capture.read()
            key_press: int = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        capture.release()

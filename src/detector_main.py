"""Main Point for the image detection"""

import os
from tkinter import Tk, filedialog

from detector import Detector

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main() -> None:
    """
    Detectron App entry point. You choose between the type of the detection and object
    """
    model_input: str = input(
        "\nPlease Chose Your Detection Model Type:\n\n"
        "\t1: Object Detection\n"
        "\t2: Keypoints Detection\n"
        "\t3: Instant Segmentation\n"
        "\t4: Instant Segmentation with Rend Point\n"
        "\t5: Panoptic Segmentation\n\n"
        "\tDetection Model Type: "
    )

    object_input: str = input(
        "\nChoose Your Object Type:\n\n input i: images or v: videos?"
        "\n\n\tObject Type: "
    )

    detect: Detector = Detector(model_type=model_input, object_type=object_input)

    root = Tk()
    root.withdraw()  # hide the root window

    # ask the user to select a file
    file_path: str = filedialog.askopenfilename()

    if object_input.lower() == "i":
        detect.detect_object_in_images(image_path=file_path)
    elif object_input.lower() == "v":
        detect.detect_object_in_video(video_path=file_path)
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

"""Device handler module"""

from typing import Literal

import torch.cuda

device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

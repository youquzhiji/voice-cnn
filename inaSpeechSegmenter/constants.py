import os
import shutil
from dataclasses import dataclass
from typing import Optional, Literal, Union

# Types
VadEngine = Literal['smn', 'sm']
PathLike = Union[str, bytes, os.PathLike]


@dataclass
class InaConfig:
    # ffmpeg command name or path
    ffmpeg: str = os.environ.get('ffmpeg', 'ffmpeg')
    # temporary directory
    tmp_dir: Optional[str] = None


ina_config = InaConfig()


# test ffmpeg installation
if shutil.which(ina_config.ffmpeg) is None:
    print("ffmpeg program not found, please install it or specify ffmpeg command path in environment variable 'ffmpeg'")
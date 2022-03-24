import os
import shutil

ffmpeg = os.environ.get('ffmpeg', 'ffmpeg')

# test ffmpeg installation
if shutil.which(ffmpeg) is None:
    print("ffmpeg program not found, please install it or specify ffmpeg command path in environment variable 'ffmpeg'")
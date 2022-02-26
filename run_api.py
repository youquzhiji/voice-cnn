# Load the API
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid

# Retrieve wav files with corresponding speaker id
root_path='../wav/' #这里是按照vox1的目录结构写的，vox1解压后的文件夹的名字是wav
wav_files=[]
import os
speaker_id = [f for f in os.listdir(root_path)]
for speaker in speaker_id:
    for dirpath, dirnames, filenames in os.walk(os.path.join(root_path,speaker)):
        wav_files+=[(speaker,os.path.join(dirpath,file)) for file in filenames]
print("start analysis in CNN model")
saved=[]
for file in wav_files:
    seg=Segmenter()
    id=file[0]
    seg_result=seg(file[1])
    saved+=[(id,each[0]) for each in seg_result]
print("analysis is ended")

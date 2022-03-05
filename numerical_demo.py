'''
A demo fof getting numerical outputs: (speaker_id,(gender,[p(female),p(male)]))
'''
# Load the API
from inaSpeechSegmenter import Segmenter
root_path='/users/yiwei/voxceleb_trainer/data/wav' #这里是按照vox1的目录结构写的，vox1解压后的文件夹的名字是wav
wav_files=[]
import os
speaker_id = [f for f in os.listdir(root_path)]
for speaker in speaker_id:
    for dirpath, dirnames, filenames in os.walk(os.path.join(root_path,speaker)):
        wav_files+=[(speaker,os.path.join(dirpath,file)) for file in filenames] #wav_file: a list of (speaker_id,speaker_audio_file)
pred=[]
for file in wav_files[:2]: # Only contains two audio files in this demo
    seg=Segmenter()
    id=file[0]
    seg_result=seg(file[1])
    pred+=[(id,each) for each in seg_result]
    # print(seg_result)
print('CNN numerical outputs:')
print(pred)
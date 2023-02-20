import os 
import random 
import torch
from torch import nn
import numpy as np
from scipy.io import wavfile
import cv2
import tempfile
import av_hubert.avhubert.utils as avhubert_utils
from av_hubert import *
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
from IPython.display import HTML
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from WavLM import WavLM, WavLMConfig

class MyDataset(Dataset):
    def __init__(self,data_path,data_label,apply_noise=False,SNR=0,is_train=False):
        self.data_path = data_path 
        self.data_label = data_label
        self.apply_noise = apply_noise
        self.SNR = SNR
        self.is_train = is_train
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self,item):
        # random.seed(1)
        data = np.load(self.data_path[item])
        audiodata =data['audio']
        videodata = data['video']
        if self.apply_noise:
            SNR = random.randint(-15,self.SNR) if self.is_train else self.SNR
            data = audiodata.astype(np.int64)
            noise = np.array(wavfile.read('/root/autodl-tmp/_background_noise_/pink_noise.wav')[1])
            start = random.randint(0,len(noise)-len(data))
            noisedata = noise[start:start+len(data)].astype(np.int64)   
            amp = (sum(data**2)) / ((10**(SNR/10))*sum(noisedata**2))
            audiodata = (data + np.sqrt(amp)*noisedata).astype(audiodata.dtype)
        
        label = self.data_label[item]        
        audio = torch.as_tensor(audiodata).to(torch.float)
        video = torch.as_tensor(videodata).to(torch.float)
        return audio,video,label

def collate_fn(batch):
    
    audio_temp = []
    video_temp = []

    audiodata,videodata,labels = tuple(zip(*batch))

    maxlen_video = max([x.shape[0] for x in videodata])
    maxlen_audio = max([len(x) for x in audiodata])

    for x in videodata:
        pad = torch.zeros((maxlen_video-x.shape[0],96,96))
        video_temp.append(torch.cat((x,pad)).unsqueeze(dim=0))
    for x in audiodata:
        pad = torch.zeros(maxlen_audio-len(x))
        audio_temp.append(torch.cat((x,pad)))

    labels = torch.as_tensor(labels)

    return torch.stack(audio_temp,dim=0),torch.stack(video_temp,dim=0),labels


class Model_V(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformerlayer = nn.TransformerEncoderLayer(768,8)
        self.transformer = nn.TransformerEncoder(self.transformerlayer,4)

        self.cls_token = nn.Parameter(torch.zeros(1,1,768))
        self.linear1 = nn.Linear(768,128)
        self.linear2 = nn.Linear(128,5)
    
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Linear(1024,768)
        self.proj1 = nn.Linear(768,768)

        utils.import_user_module(Namespace(user_dir='/root'))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(['/root/finetune-model.pt'])
        video_model = models[0].encoder.w2v_model

        
    def forward(self,Video,video_model):
        with torch.no_grad():
            video_feature =  video_model.extract_finetune(source={'video': Video, 'audio': None}, padding_mask=None, output_layer=None)[0].permute(1,0,2)

        Y = video_feature
        cls_token = self.cls_token.expand(-1,Video.shape[0],-1)
        Y = torch.cat((cls_token,Y))
        Y = self.transformer(Y)
        Y = Y[0,:,:]
        Y = self.dropout(self.relu(self.linear1(Y)))
        Y = self.linear2(Y)
        
        return Y
    

class Model_A(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transformerlayer = nn.TransformerEncoderLayer(1024,8)
        self.transformer = nn.TransformerEncoder(self.transformerlayer,4)

        self.cls_token = nn.Parameter(torch.zeros(1,1,1024))
        self.linear1 = nn.Linear(1024,128)
        self.linear2 = nn.Linear(128,5)
    
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self,Audio,audio_feature_extracter):
        with torch.no_grad():
            audio_feature = audio_feature_extracter.extract_features(Audio)[0].permute(1,0,2)
        Y = audio_feature
        cls_token = self.cls_token.expand(-1,Audio.shape[0],-1)
        Y = torch.cat((cls_token,Y))
        Y = self.transformer(Y)
        Y = Y[0,:,:]
        Y = self.dropout(self.relu(self.linear1(Y)))
        Y = self.linear2(Y)
        
        return Y
    

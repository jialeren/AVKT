import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
from utils import MyDataset, collate_fn
from torch import nn


# KWS_V = torch.load
# KWS_AV = torch.load


def read_data(rootpath='/root/autodl-tmp/PKUTEST', val_rate=0.1):
    random.seed(0)
    assert os.path.exists(rootpath), "path doesn't exist"

    keyword_dict = {'00001': 0, '00010': 1, '00100': 2, '01000': 3, '10000': 4}

    train_data_path = []
    train_data_label = []
    val_data_path = []
    val_data_label = []

    path_list = []
    for file in os.listdir(rootpath):
        if keyword_dict.__contains__(file[:5]):
            path_list.append(os.path.join(rootpath, file))

    val_path = random.sample(path_list, k=int(len(path_list) * val_rate))
    num_every_class = {key: 0 for key in keyword_dict.keys()}

    for path in path_list:
        num_every_class[path.split('/')[-1][:5]] += 1
        if path in val_path:  # 如果在验证集中，则放入验证集
            val_data_path.append(path)
            val_data_label.append(keyword_dict[path.split('/')[-1][:5]])
        else:
            train_data_path.append(path)
            train_data_label.append(keyword_dict[path.split('/')[-1][:5]])

    print(num_every_class)
    print(f'数据集中共包含{len(path_list)}个音频样本')
    return train_data_path, train_data_label, val_data_path, val_data_label


test_data_path, test_data_label, _, __ = read_data(rootpath='/root/autodl-tmp/PKUTEST',
                                                   val_rate=0)  # 取LRS2TEST里所有样本作为测试集

test_dataset = MyDataset(test_data_path, test_data_label)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=16, collate_fn=collate_fn)

device = 'cuda'
loss = nn.CrossEntropyLoss()



checkpoint = torch.load('/root/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
audio_feature_extracter = WavLM(cfg).to(device)
audio_feature_extracter.load_state_dict(checkpoint['model'])
KWS_A = torch.load('/root/PKU_A.pt')

KWS_A.eval()

with torch.no_grad():
    for SNR in [10,5,0,-5,-10]:
        test_dataset = MyDataset(test_data_path,test_data_label,apply_noise=True,SNR=SNR)
        test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=16,collate_fn=collate_fn)
        acc = []
        for i in range(2):
            test_accuracy = 0
            for data in test_dataloader:
                audio, video, y = data
                audio,  y = audio.to(device),y.to(device)
                y_hat = KWS_A(audio,audio_feature_extracter)
                lll = loss(y_hat,y)
                accuracy = (y_hat.argmax(1) == y).sum()
                test_accuracy += accuracy
            acc.append(test_accuracy.item()/len(test_dataset))
        print(acc)
        print("SNR:{}dB,测试集acc:{}".format(SNR, sum(acc)/len(acc)))


class Model_AV(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(10, 5)
        self.sacle = nn.Linear(10, 1)

    def forward(self, Audio, Video):
        A = KWS_A(Audio, audio_feature_extracter)
        V = KWS_V(Video=Video, video_model=video_model)

        Y = torch.cat((A, V), dim=1)
        Y = self.linear(Y)
        return Y


KWS_AV = torch.load('/root/AV3.pt')
KWS_AV.eval()
with torch.no_grad():
    for SNR in [-5, -10]:
        test_dataset = MyDataset(test_data_path, test_data_label, apply_noise=True, SNR=SNR)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=collate_fn)
        acc = []
        for i in range(5):
            test_accuracy = 0
            for data in test_dataloader:
                audio, video, y = data
                audio, video, y = audio.to(device), video.to(device), y.to(device)
                y_hat = KWS_AV(audio, video)
                lll = loss(y_hat, y)
                accuracy = (y_hat.argmax(1) == y).sum()
                test_accuracy += accuracy
            acc.append(test_accuracy.item() / len(test_dataset))
        print(acc)
        print("SNR:{}dB,测试集acc:{}".format(SNR, sum(acc) / len(acc)))
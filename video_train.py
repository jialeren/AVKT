import random
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch import nn
% cd
av_hubert / avhubert
from torch.utils.tensorboard import SummaryWriter
from scipy.io import wavfile
import numpy as np
import os
import random

from utils import *


def read_data(rootpath='/root/autodl-tmp/PKU-train', val_rate=0.1):
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


train_data_path, train_data_label, val_data_path, val_data_label = read_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32

train_dataset = MyDataset(train_data_path, train_data_label, apply_noise=True, is_train=True)
test_dataset = MyDataset(val_data_path, val_data_label)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)

loss = nn.CrossEntropyLoss()

# KWS_V = Model_V().to(device)
KWS_V = torch.load('/root/PKU_V.pt')

# checkpoint = torch.load('/root/WavLM-Large.pt')
# cfg = WavLMConfig(checkpoint['cfg'])
# audio_feature_extracter = WavLM(cfg).to(device)
# audio_feature_extracter.load_state_dict(checkpoint['model'])
# KWS_A = Model_A().to(device)

utils.import_user_module(Namespace(user_dir='/root'))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(['/root/finetune-model.pt'])
video_model = models[0].encoder.w2v_model.to(device)

# KWS_A = torch.load('/root/LRS_A.pt')

writer = SummaryWriter('/root/tf-logs')
loss = nn.CrossEntropyLoss()
loss.to(device)

num_epoch = 35
total_train_step = 0

print('**********开始训练0223***  on', device, '************')

optimizer = torch.optim.SGD(KWS_V.parameters(), lr=1e-2)

for i in range(num_epoch):
    if i > 30:
        optimizer = torch.optim.SGD(KWS_V.parameters(), lr=1e-4)
    elif i > 25:
        optimizer = torch.optim.SGD(KWS_V.parameters(), lr=1e-3)
    print(f'******epoch:{i + 1}*******')
    KWS_V.train()
    train_accuracy = 0
    for data in train_dataloader:
        audio, video, y = data
        video, y = video.to(device), y.to(device)
        y_hat = KWS_V(video, video_model)
        ll = loss(y_hat, y)
        # print(ll)
        optimizer.zero_grad()
        ll.backward()
        optimizer.step()

        accuracy = (y_hat.argmax(1) == y).sum()
        train_accuracy += accuracy

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f'训练次数{total_train_step},loss={ll.item()}')
        writer.add_scalar('Loss8/train_loss', ll.item(), total_train_step / 100)
    acc0 = train_accuracy / len(train_dataset)
    print("训练集acc:{}".format(acc0))

    KWS_V.eval()
    test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            audio, video, y = data
            video, y = video.to(device), y.to(device)
            y_hat = KWS_V(video, video_model)
            lll = loss(y_hat, y)
            accuracy = (y_hat.argmax(1) == y).sum()
            test_accuracy += accuracy
        writer.add_scalar('Loss8/test_loss', lll.item(), i)
        acc = test_accuracy / len(test_dataset)
        print("测试集acc:{}".format(acc))
        # bestacc = 0.85
        # if acc> 0.85:
        #     if acc > bestacc:
        #         torch.save(KWS_V,'/root/PKU_V.pt')
        #         bestacc = acc

#     writer.add_scalar('Acc8/train_acc', acc0, i)
#     writer.add_scalar('Acc8/test_acc', acc, i)






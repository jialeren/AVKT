#### Paper: Audio-Visual Keyword Transformer for Unconstrained Sentence-Level Keyword Spotting

Contributing to *CAAI Transactions on Intelligence Technology*

#### 1.Requirements 

In our method, two pre-trained model **WavLm** and **AV-HuBERT** are used for feature extractor. So you're supposed to deploy both models according to their respective documentation：

WavLM:https://github.com/microsoft/unilm

AV-HuBERT:https://github.com/facebookresearch/av_hubert

For other enviroment requirements, Set up python 3.6.7 environment：

```
pytorch==1.4.0
torchvision==0.5.0
numpy==1.19.3
opencv==4.5.0

Other dependencies: 
scikit-image, scipy, tqdm,os, sys...
```

#### 2.Pre-processing 

- ##### Dataset available

The PKU-KWS dataset is available at https://zenodo.org/record/6792058 

The LRS2 dataset is available at https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html  by *Visual Geometry Group of Oxford University*.

You can download these two dataset and arrange them in the format required for data preprocessing.

- ##### Processing

After preparing for dataset, you can run python `video_processing.py` to pre-process the dataset. The label can be generate from filename with the following relationship:

00001 - Xiexie - When  |  00010 - Nihao - Why | 00100 - Dazhe - Very | 01000 - Jiezhang- One | 10000 - Pengpeng - About 

#### 3.Training

- ​	First, you can run python `audio_train.py` to train an audio model and get `A.pt`

- ​	Then, you can run python `audio_train.py` to train a video model and get `V.pt`

- ​	Now, you can run python `audio_visual_train.py` to train a audio_visual model and get `AV.pt` for audio-visual test.

​	*Note: This is only the process of training a five-category model on the PKU-KWS dataset. Changing to LRS2 or to other classification tasks only requires a simple modification in the DataLoader part of the code.

 #### 4.Test 

​	You can run python `test.py` for test. 


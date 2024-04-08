# FAME'24 Challenge (Face-voice Association in Multilingual Environments 2024) (Open)
For more information on challenge please visit [Challenge Webpage](https://mavceleb.github.io/dataset/index.html)
# Baseline
Baseline code for v2 of MAV-Celeb dataset based on _'Fusion and Orthogonal Projection for Improved Face-Voice Association'_ [{paper}](https://ieeexplore.ieee.org/abstract/document/9747704) [{code}](https://github.com/msaadsaeed/FOP)
## Task
Face-voice association is established in cross-modal verification task. The goal of the cross-modal verification task is to verify if, in a given single sample with both a face and voice, both belong to the same identity. In addition, we analyze the impact of multiple of languages on cross-modal verification task.
<p align='center'>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/challenge_task_diag_r.jpg' width=70% height=70%  style="border-radius: 25px">
</p>

<table border="1" align='center'>
  <tr>
    <td colspan="4" align="center" ><b>V2-EH</b></td>
  </tr>
  <tr>
    <td>Method</td>
    <td>Configuration</td>
    <td align='center'>English test<br>(EER)</td>
    <td align='center'>Hindi test<br>(EER)</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">FOP</td>
    <td>English train</td>
    <td><b>20.8</b></td>
    <td>24.0</td>
  </tr>
  <tr>
    <td>Hindi train</td>
    <td>24.0</td>
    <td><b>19.3</b></td>
  </tr>
  <tr>
    <td colspan="4" align="center"><b>V1-EU</b></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td align='center'>English test<br>(EER)</td>
    <td align='center'>Urdu test<br>(EER)</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">FOP</td>
    <td>English train</td>
    <td><b>29.3</b></td>
    <td>37.9</td>
  </tr>
  <tr>
    <td>Urdu train</td>
    <td>40.4</td>
    <td><b>25.8</b></td>
  </tr>
</table>



## Evaluation Protocol
The aim is to study the impact of language on speaker recognition and face-voice assoication methods. For this we train a model X on one language (English) then test on same language (English) and unheard language (Hindi). Similarly we train a model Y on one language (Hindi) then test the model on same language (Hindi) and unheard language (English) as shown in figure below. It is also important to note that the test identities are also unheard by the network meaning the test set is disjoint from the train network. For example: v2 has 84 identities both having English and Hindi voice samples. We have separated 6 identities for test set while leverage reamining for training the model.<br>

<p align='center'>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/eng_heard.JPG' width=40% height=40%>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/hin_heard.JPG' width=40% height=40%>
</p>


## Extracted features:
### Face Features:
For Face Embeddings (4096-D) we use [VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/). The model can be downloaded [here](https://drive.google.com/drive/folders/1ct_TXo2x-1tKGAnGYDaC6XzIPRaVN6J-?usp=sharing). Run `vggFaceFeat.py` for face feature extraction.

### Voice Features:
For Voice Embeddings (512-D) we use the method described in [Utterance Level Aggregator](https://arxiv.org/abs/1902.10107). The code we used is released by authors and is [publicly available](https://github.com/WeidiXie/VGG-Speaker-Recognition). We fine tuned the model on v2 split of MAV-Celeb dataset for feature extraction. The pre-trained model on MAV-Celeb ( v2) can be downloaded [here](https://drive.google.com/drive/folders/1ykJ3rAPLN0x1n5nVaw3QVPi9vZXlrfe6?usp=sharing). Run `uttLevelVoiceFeat.py` for voice feature extraction.

Pre extracted features for reproducing the baseline results can be downloaded [here](https://drive.google.com/drive/folders/1TYxRAMzzn0ZO9pYTXYlhc67YGzvXhMV1?usp=sharing).

## Splits and Raw Data
The raw data and train/test splits can be downloaded [here.](https://drive.google.com/drive/folders/1X8q4LjpII3G6sIexOiBZDd_4hTaxTjto?usp=sharing)

#### Submission

We provide both train and test splits with ground truth for v2 of MAV-Celeb dataset. Participants can use this split for fine-tuning their method. However, for v1 the test files are in format as below:
```
ysuvkz41 voices/English/00000.wav faces/English/00000.jpg 
tog3zj45 voices/English/00001.wav faces/English/00001.jpg 
ky5xfj1d voices/English/00002.wav faces/English/00002.jpg 
yx4nfa35 voices/English/01062.wav faces/English/01062.jpg 
bowsaf5e voices/English/01063.wav faces/English/01063.jpg 
```

We have kept the ground truth for fair evaluation during FAME challenge. Participants are expected to compute and submit a text file including the `id` and `L2 Scores` in the following format:
```
ysuvkz41 0.9988
tog3zj45 0.1146
ky5xfj1d 0.6514
yx4nfa35 1.5321
bowsaf5e 1.6578
```

## Hierarchy:
```
├── feats
│ ├── v2
│ │ ├── English_feats
│ │ │ ├── .csv and .txt files
│ │ ├── Hindi_feats
│ │ │ ├── .csv and .txt files
├── v2_models
│ ├── English_fop_model
│ │ ├── checkpoints.pth.tar
│ ├── Hindi_fop_model
│ │ ├── checkpoints.pth.tar
├── main.py
├── online_evaluation.py
├── retrieval.py
├── test.py
```

# Setup
We have used anaconda for setting up the environemnt for our experiments:
```
python==3.6.5
```
[CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) Setup:

* CUDA Toolkit 10.2
* cudnn v8.2.1.32 for CUDA10.2

To install PyTorch with GPU support:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```


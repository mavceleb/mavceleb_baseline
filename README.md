# FAME'26 Challenge (Face-voice Association in Multilingual Environments 2026)

## $\textcolor{green}{Accepting \space Submissions!}$

For more information on challenge please see [evaluation plan]((https://arxiv.org/abs/2508.04592)
[Challenge Webpage](https://mavceleb.github.io/dataset/index.html)

# Baseline

Baseline code for v2 & v3 of MAV-Celeb dataset based on _'Fusion and Orthogonal Projection for Improved Face-Voice Association'_ [{paper}](https://ieeexplore.ieee.org/abstract/document/9747704) [{code}](https://github.com/msaadsaeed/FOP)

## Task

Face-voice association is established in cross-modal verification task. The goal of the cross-modal verification task is to verify if, in a given single sample with both a face and voice, both belong to the same identity. In addition, we analyze the impact of multiple of languages on cross-modal verification task.

<p align='center'>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/challenge_task_diag_r.jpg' style="border-radius: 25px">
</p>

<table border="1" align='center'>
  <tr>
    <td colspan="4" align="center"><b>MAV-Celeb v1 (EU)</b></td>
  </tr>
  <tr>
    <td>Method</td>
    <td>Configuration</td>
    <td align='center'>English test<br>(EER)</td>
    <td align='center'>Urdu test<br>(EER)</td>
    <td align='center'></td>
  </tr>
  <tr>
    <td rowspan="2" align="center">FOP</td>
    <td>English train</td>
    <td>45.1</td>
    <td>48.3</td>
    <td rowspan="2" align="center">46.2</td>
  </tr>
  <tr>
    <td>Urdu train</td>
    <td>47.0</td>
    <td>44.3</td>
  </tr>
  
  <tr>
    <td colspan="4" align="center"><b>MAV-Celeb v2 (EH)</b></td>
  </tr>
  <tr>
    <td>Method</td>
    <td>Configuration</td>
    <td align='center'>English test<br>(EER)</td>
    <td align='center'>Hindi test<br>(EER)</td>
    <td align='center'></td>

  </tr>
  <tr>
    <td rowspan="2" align="center">FOP</td>
    <td>English train</td>
    <td>35.7</td>
    <td>36.7</td>
    <td rowspan="2" align="center">37.2</td>

  </tr>
  <tr>
    <td>Hindi train</td>
    <td>38.9</td>
    <td>37.3</td>
  </tr>
  
  <tr>
    <td colspan="4" align="center"><b>MAV-Celeb V3 (EG)</b></td>
  </tr>
  <tr>
    <td>Method</td>
    <td>Configuration</td>
    <td align='center'>English test<br>(EER)</td>
    <td align='center'>German test<br>(EER)</td>
    <td align='center'></td>
  </tr>
  <tr>
    <td rowspan="2" align="center">FOP</td>
    <td>English train</td>
    <td>34.5</td>
    <td>43.7</td>
    <td rowspan="2" align="center">40.2</td>
  </tr>
  <tr>
    <td>German train</td>
    <td>43.2</td>
    <td>39.6</td>
  </tr>
</table>

## Evaluation Protocol

The aim is to study the impact of language on face-voice assoication methods. For this we train a model X on one language (English) then test on same language (English) and unheard language (Hindi). Similarly we train a model Y on one language (Hindi) then test the model on same language (Hindi) and unheard language (English) as shown in figure below. It is also important to note that the test identities are also unheard by the network meaning the test set is disjoint from the train network. For example: v2 has 84 identities both having English and Hindi voice samples. We have separated 6 identities for test set while leverage reamining for training the model.<br>

<p align='center'>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/eng_heard.JPG' width=40%>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/hin_heard.JPG' width=40%>
</p>

## Extracted features:

### Face Features:

For Face Embeddings (4096-D) we use [VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/). The model can be downloaded [here](https://drive.google.com/drive/folders/1ct_TXo2x-1tKGAnGYDaC6XzIPRaVN6J-?usp=sharing). Run `vggFaceFeat.py` for face feature extraction.

### Voice Features:

For Voice Embeddings (512-D) we use the method described in [Utterance Level Aggregator](https://arxiv.org/abs/1902.10107). The code we used is released by authors and is [publicly available](https://github.com/WeidiXie/VGG-Speaker-Recognition). We fine tuned the model on v1, v2 and v3 split of MAV-Celeb dataset for feature extraction. The pre-trained model on MAV-Celeb (v1, v2, v3) can be downloaded [here](https://drive.google.com/drive/folders/1ykJ3rAPLN0x1n5nVaw3QVPi9vZXlrfe6?usp=sharing). Run `uttLevelVoiceFeat.py` for voice feature extraction.

Pre extracted features for reproducing the baseline results can be downloaded. You can download v1, v2 and v3 feature files of faces and voices [here.](https://drive.google.com/drive/folders/1LfCxZiAqmsD9sgEMRrJgN5QBr_CL-hzD?usp=sharing)

## Splits and Raw Data

Download [raw data](https://drive.google.com/drive/folders/1OJyjXJULErvrvzLQmpJn5v8rRo0n_fod?usp=sharing) and [train/test splits](https://drive.google.com/drive/folders/1MEHtEVh9lSa9hNZxjEfNJnE3qrpm_PKw?usp=sharing)

#### Submission

We provide both train and test splits of MAV-Celeb dataset. For v1, v2, v3, the test files are in format as below:

```
ysuvkz41 voices/English/00000.wav faces/English/00000.jpg
tog3zj45 voices/English/00001.wav faces/English/00001.jpg
ky5xfj1d voices/English/00002.wav faces/English/00002.jpg
yx4nfa35 voices/English/01062.wav faces/English/01062.jpg
bowsaf5e voices/English/01063.wav faces/English/01063.jpg
```

We have kept the ground truth for fair evaluation during FAME challenge. Participants are expected to compute and submit a text file including the `id` and `L2 Scores` in the following format. We also provide `computeScore.py` to generate sample submission score files. Set `ver` and `heard_lang` accordingly and run the script. For example, if `ver='v1'` and `heard_lang='English'`, the script will generate two files in the given format: `sub_scores_English_heard.txt` and `sub_scores_English_unheard.txt`. The heard scores are for `English` test language whereas unheard scores for language that is unheard by the model i.e. `Urdu`.

```
ysuvkz41 0.9988
tog3zj45 0.1146
ky5xfj1d 0.6514
yx4nfa35 1.5321
bowsaf5e 1.6578
```

Link to Codalab: [Codalab/Codabench](https://www.codabench.org/competitions/9467/)

## Hierarchy:

```
├── dataset
│ ├── .zip files
├── preExtracted_vggFace_utteranceLevel_Features
│ ├── v1
│ │ ├── Urdu
│ │ │ ├── .csv and .txt files
│ ├── v2
│ │ ├── Hindi
│ │ │ ├── .csv and .txt files
│ ├── v3
│ │ ├── German
│ │ │ ├── .csv and .txt files
├── face_voice_association_splits
│ ├── v1
│ │ ├── .txt split files
│ ├── v2
│ │ ├── .txt split files
│ ├── v3
│ │ ├── .txt split files
├── v1_models
│ ├── Urdu_fop_model
│ │ ├── checkpoints.pth.tar
│ ├── English_fop_model
│ │ ├── checkpoints.pth.tar
├── main.py
├── computeScore.py
├── retrieval.py
```

# Setup

We have used anaconda for setting up the environemnt for our experiments:

```
python==3.6.5
```

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) Setup:

- CUDA Toolkit 10.2
- cudnn v8.2.1.32 for CUDA10.2

To install PyTorch with GPU support:

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

$\textcolor{red}{Last \space Year \space Results \space}$

| Rank | Team Name    | Primary Contact | Affiliation                               | Score (EER) | System Description Report                                                                                |
| ---- | ------------ | --------------- | ----------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------- |
| 1    | HLT          | Tao Ruijie      | National University of Singapore          | 19.91       | [Click Here](https://github.com/mavceleb/mavceleb_baseline/blob/main/description_files/1hlt.pdf)         |
| 2    | Audio_Visual | Wuyang Chen     | National University of Defense Technology | 20.51       | [Click Here](https://github.com/mavceleb/mavceleb_baseline/blob/main/description_files/2audioVisual.pdf) |
| 3    | Xaiofei      | Tang Jie Hui    | Hefei University of Technology            | 21.76       | [Click Here](https://github.com/mavceleb/mavceleb_baseline/blob/main/description_files/3xaiofei.pdf)     |

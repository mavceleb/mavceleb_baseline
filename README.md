# FAME'24 Challenge (Face-voice Association in Multilingual Environments 2024)
For more information on challenge please visit [Challenge Webpage](https://mavceleb.github.io/dataset/index.html)
# Baseline
Baseline code for v2 of MAV-Celeb dataset based on _'Fusion and Orthogonal Projection for Improved Face-Voice Association'_ [{paper}](https://ieeexplore.ieee.org/abstract/document/9747704) [{code}](https://github.com/msaadsaeed/FOP)
### Task
Face-voice association is established in cross-modal verification task. The goal of the cross-modal verification task is to verify if, in a given single sample with both a face and voice, both belong to the same identity. In addition, we analyze the impact of multiple of languages on cross-modal verification task.
<p align='center'>
  <img src='https://github.com/mavceleb/mavceleb_baseline/blob/main/images/challenge_task_diag_r.jpg' width=70% height=70%>
</p>

### Setup
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

## Extracted features:
### Face Features:
For Face Embeddings (4096-D) we use [VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/). The model can be downloaded [here](https://drive.google.com/drive/folders/1ct_TXo2x-1tKGAnGYDaC6XzIPRaVN6J-?usp=sharing). Run `vggFaceFeat.py` for face feature extraction.

### Voice Features:
For Voice Embeddings (512-D) we use the method described in [Utterance Level Aggregator](https://arxiv.org/abs/1902.10107). The code we used is released by authors and is [publicly available](https://github.com/WeidiXie/VGG-Speaker-Recognition). We fine tuned the model on v2 split of MAV-Celeb dataset for feature extraction. The pre-trained model on MAV-Celeb ( v2) can be downloaded [here](https://drive.google.com/drive/folders/1ykJ3rAPLN0x1n5nVaw3QVPi9vZXlrfe6?usp=sharing). Run `uttLevelVoiceFeat.py` for voice feature extraction.

Pre extracted features can be downloaded [here](https://drive.google.com/drive/folders/1TYxRAMzzn0ZO9pYTXYlhc67YGzvXhMV1?usp=sharing).

## Hierarchy:
├── feats/<br>
│ ├── v2/<br>
│ │ ├── English_feats<br>
│ │ │ ├── .csv and .txt files<br>
│ │ ├── Hindi_feats<br>
│ │ │ ├── .csv and .txt files<br>
├── v2_models/<br>
│ ├── English_fop_model/<br>
│ │ ├── checkpoints.pth.tar<br>
│ ├── Hindi_fop_model<br>
│ │ ├── checkpoints.pth.tar<br>
├── main.py/<br>
├── online_evaluation.py/<br>
├── retrieval.py/<br>
├── test.py/<br>



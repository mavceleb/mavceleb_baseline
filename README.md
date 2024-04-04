
Download Feats [here](https://drive.google.com/drive/folders/1TYxRAMzzn0ZO9pYTXYlhc67YGzvXhMV1?usp=sharing).
## Extracted features:
### Face Features:
For Face Embeddings (4096-D) we use [VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/). We use the Keras implementation of this paper from [this repository](https://gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9)
### Voice Features:
For Voice Embeddings (512-D) we use the method described in [Utterance Level Aggregator](https://arxiv.org/abs/1902.10107). The code we used is released by authors and is [publicly available](https://github.com/WeidiXie/VGG-Speaker-Recognition). We fine tuned the model on v2 split of MAV-Celeb dataset for feature extraction.
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



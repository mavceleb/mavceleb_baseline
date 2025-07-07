
import model
import argparse
import os
import utils as ut
import numpy as np
from glob import glob
import pickle
import progressbar

import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='0', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
# set up version of the model.
parser.add_argument('--version', default='v1', choices=['v1', 'v2', 'v3'], type=str)

global args
args = parser.parse_args()


n_classes = 64 if args.version == 'v1' else 78 if args.version == 'v2' else 50

params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': n_classes,
              'sampling_rate': 16000,
              'normalize': True,
              }

network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

path_to_model = 'weights.h5'

if os.path.isfile(path_to_model):
    network_eval.load_weights(path_to_model, by_name=True)
else:
    print('Error-File not Found')

wav = 'path to .wav file'
specs = ut.load_data(wav, win_length=params['win_length'], sr=params['sampling_rate'],
          hop_length=params['hop_length'], n_fft=params['nfft' ],
          spec_len=params['spec_len'], mode='eval')
specs = np.expand_dims(np.expand_dims(specs, 0), -1)
feat = network_eval.predict(specs)


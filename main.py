
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
from scipy import random
from sklearn import preprocessing
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

# In[0]

def read_data(FLAGS):
    train_file_face = '../feats/%s/%s_feats/%s_faces_train.csv'%(FLAGS.ver, FLAGS.train_lang, FLAGS.train_lang)
    train_file_voice = '../feats/%s/%s_feats/%s_voices_train.csv'%(FLAGS.ver, FLAGS.train_lang, FLAGS.train_lang)
    
    print('Reading Train Faces')
    img_train = pd.read_csv(train_file_face, header=None)
    train_label = img_train[-1]
    img_train = np.asarray(img_train)
    img_train = img_train[:, :-1]
    train_label = np.asarray(train_label)
    print('Reading Voices')
    voice_train = pd.read_csv(train_file_voice, header=None)
    voice_train = np.asarray(voice_train)
    voice_train = voice_train[:, :-1]
    
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    print("Train file length", len(img_train))
        
    print('Shuffling\n')
    combined = list(zip(img_train, voice_train, train_label))
    img_train = []
    voice_train = []
    train_label = []
    random.shuffle(combined)
    img_train[:], voice_train, train_label[:] = zip(*combined)
    combined = [] 
    img_train = np.asarray(img_train).astype(np.float)
    voice_train = np.asarray(voice_train).astype(np.float)
    train_label = np.asarray(train_label)
    
    
    return img_train, voice_train, train_label


# In[1]
print('Training')
from retrieval_model import FOP

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
 
def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(face_train, voice_train, train_label):
    
    n_class = 64 if FLAGS.ver == 'v1' else 78
    model = FOP(FLAGS, face_train.shape[1], voice_train.shape[1], n_class)
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss().cuda()
    opl_loss = OrthogonalProjectionLoss().cuda()
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True
    
# =============================================================================
#     For Linear Fusion
# =============================================================================
    
    if FLAGS.fusion == 'linear':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                        {'params' : model.fusion_layer.weight1},
                        {'params' : model.fusion_layer.weight2}]
    
    
# =============================================================================
#     For Gated Fusion
# =============================================================================
    
    elif FLAGS.fusion == 'gated':
    
        parameters = [
                      {'params' : model.face_branch.fc1.parameters()},
                      {'params' : model.voice_branch.fc1.parameters()},
                      {'params': model.logits_layer.parameters()},
                      {'params' : model.fusion_layer.attention.parameters()}]

    optimizer = optim.Adam(parameters, lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    epoch=1
    num_of_batches = (len(train_label) // FLAGS.batch_size)
    
    
    save_dir = '%s_%s_%s_alpha_%0.2f'%(FLAGS.ver, FLAGS.train_lang, FLAGS.fusion, FLAGS.alpha)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    while (epoch < FLAGS.epochs):
        loss_per_epoch = 0
        loss_plot = []
        print('Epoch %03d'%(epoch))
        for idx in tqdm(range(num_of_batches)):
            face_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, face_train)
            voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
            loss_tmp, loss_opl, loss_soft, s_fac, d_fac = train(face_feats, voice_feats, 
                                                         batch_labels, 
                                                         model, optimizer, ce_loss, opl_loss, FLAGS.alpha)
            loss_per_epoch+=loss_tmp
        
        loss_per_epoch/=num_of_batches
        
        loss_plot.append(loss_per_epoch)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d_%0.3f.pth.tar'%(epoch, loss_per_epoch))

        print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f, '%(epoch, FLAGS.epochs, loss_per_epoch, FLAGS.alpha))
        
        loss_per_epoch = 0
        epoch += 1
            
    return
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def train(face_feats, voice_feats, labels, model, optimizer, ce_loss, opl_loss, alpha):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    face_feats = torch.from_numpy(face_feats).float()
    voice_feats = torch.from_numpy(voice_feats).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        face_feats, voice_feats, labels = face_feats.cuda(), voice_feats.cuda(), labels.cuda()

    face_feats, voice_feats, labels = Variable(face_feats), Variable(voice_feats), Variable(labels)
    comb, face_embeds, voice_embeds = model.train_forward(face_feats, voice_feats, labels)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)
    
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average
 
def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--ver', default='v1', type=str, help='Dataset version')
    parser.add_argument('--train_lang', default='Urdu', type=str, help='Training language')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Max number of epochs to train, number')
    parser.add_argument('--alpha', type=list, default=1, help='Alpha Values List')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
        
    face_train, voice_train, train_label = read_data(FLAGS)
    main(face_train, voice_train, train_label)

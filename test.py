
import argparse
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from retrieval_model import FOP
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

ver = 'v2'
heard_lang = 'Hindi'

if (ver == 'v1' and heard_lang == 'Hindi') or (ver == 'v2' and heard_lang == 'Urdu'):
    raise ValueError("Contradictory combination: ver={} and heard_lang={}".format(ver, heard_lang))

assert ver == 'v1' or ver == 'v2', f"Invalid value for ver: {ver}"
assert heard_lang == 'Urdu' or heard_lang == 'Hindi' or heard_lang == 'English', f"Invalid value for lang: {heard_lang}"

if ver == 'v1':
    unheard_lang = 'Urdu' if heard_lang == 'English' else 'English'
if ver == 'v2':
    unheard_lang = 'Hindi' if heard_lang == 'English' else 'English'

print('Heard_Language: %s'%(heard_lang))
print('Unheard Language: %s'%(unheard_lang))

def read_data(ver, test_file_face, test_file_voice):
    print('Reading Test Face')
    face_test = pd.read_csv(test_file_face, header=None)
    print('Reading Test Voice')
    voice_test = pd.read_csv(test_file_voice, header=None)
    
    face_test = np.asarray(face_test)
    face_test = face_test[:, :4096]
    voice_test = np.asarray(voice_test)
    voice_test = voice_test[:, :512]
    
    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    return face_test, voice_test

def test(face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard):
    
    n_class = 64 if ver == 'v1' else 78
    model = FOP(FLAGS, face_test_heard.shape[1], voice_test_heard.shape[1], n_class)
    checkpoint = torch.load(FLAGS.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format('checkpoint.pth.tar', checkpoint['epoch']))
    model.eval()
    model.cuda()
    
    if FLAGS.cuda:
        face_test_heard, voice_test_heard = face_test_heard.cuda(), voice_test_heard.cuda()
        face_test_unheard, voice_test_unheard = face_test_unheard.cuda(), voice_test_unheard.cuda()

    face_test_heard, voice_test_heard = Variable(face_test_heard), Variable(voice_test_heard)
    face_test_unheard, voice_test_unheard = Variable(face_test_unheard), Variable(voice_test_unheard)
    print('Computing scores')
    with torch.no_grad():
        _, face_heard, voice_heard = model(face_test_heard, voice_test_heard)
        _, face_unheard, voice_unheard = model(face_test_unheard, voice_test_unheard)
                
        face_heard, voice_heard = face_heard.data, voice_heard.data
        face_unheard, voice_unheard = face_unheard.data, voice_unheard.data
        
        face_heard, voice_heard = face_heard.cpu().detach().numpy(), voice_heard.cpu().detach().numpy()
        face_unheard, voice_unheard = face_unheard.cpu().detach().numpy(), voice_unheard.cpu().detach().numpy()
        
        scores_heard = np.linalg.norm(face_heard - voice_heard, axis=1, keepdims=True)
        scores_unheard = np.linalg.norm(face_unheard - voice_unheard, axis=1, keepdims=True)
        
        gt_heard = []
        gt_unheard = []
    
        with open('./preExtracted_vggFace_utteranceLevel_Features/%s/%s_test.txt'%(ver, heard_lang)) as f:
            for dat in f:
                dat = dat.split(' ')[1]
                gt_heard.append(int(dat))
        
        with open('./preExtracted_vggFace_utteranceLevel_Features/%s/%s_test.txt'%(ver, unheard_lang)) as f:
            for dat in f:
                dat = dat.split(' ')[1]
                gt_unheard.append(int(dat))
        
        fpr, tpr, thresholds = metrics.roc_curve(gt_heard, -scores_heard)
        eer_heard = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        fpr, tpr, thresholds = metrics.roc_curve(gt_unheard, -scores_unheard)
        eer_unheard = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        print('%s to %s EER (Heard): %0.3f'%(heard_lang, heard_lang, eer_heard))
        print('%s to %s EER (Heard): %0.3f'%(heard_lang, unheard_lang, eer_unheard))
        
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training')
    parser.add_argument('--ckpt', type=str, default='./%s_models/%s_fop_model/checkpoint.pth.tar'%(ver, heard_lang), help='Checkpoints directory.')
    
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.cuda = torch.cuda.is_available()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    print('Loading Heard Language Data')
    test_file_face = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_faces_test.csv'%(ver, heard_lang, heard_lang)
    test_file_voice = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_voices_test.csv'%(ver, heard_lang, heard_lang)
    face_test_heard, voice_test_heard = read_data(ver, test_file_face, test_file_voice)
    print('Loading UnHeard Language Data')
    test_file_face = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_faces_test.csv'%(ver, heard_lang, unheard_lang)
    test_file_voice = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_voices_test.csv'%(ver, heard_lang, unheard_lang)
    face_test_unheard, voice_test_unheard = read_data(ver, test_file_face, test_file_voice)
    test(face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard)
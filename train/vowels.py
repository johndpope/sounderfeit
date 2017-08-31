
__all__ = ['sample_random', 'sample_params',
           'data_domain', 'param_domain'
           'encode_data', 'decode_data',
           'encode_params', 'decode_params',
           'domain_size', 'name']

import h5py
import numpy as np
from scipy.signal import lfilter

name = 'vowels'
mode = 'diffed'
# mode = 'periods'

randomize_phase = False
randomize_polarity = False

# make a memory-resident copy
with h5py.File('vowels.h5', mode='r') as h:
    data = np.copy(h[mode])
    stds = np.copy(h['stds'])
    means = np.copy(h['means'])

random_indexes = np.arange(data.shape[0])
np.random.shuffle(random_indexes)
testing_indexes = random_indexes[:len(random_indexes)//4]
training_indexes = random_indexes[len(random_indexes)//4:]

y_index = slice(0,1) # vowel number (0..4 inclusive for a-e-i-o-u)

def sample_random(N,mode='train'):
    global sample_indexes, sample_indexes_pos
    if N<=0: return np.array([]), np.array([])
    i = np.array([],dtype=int)
    d = data[{'train':training_indexes,
              'train_paramloss':training_indexes,
              'test':testing_indexes,
              None:slice(None)}[mode], :]
    i = np.arange(d.shape[0])
    np.random.shuffle(i)
    i = i[:N]
    # # h5py can't handle duplicate indexes
    # i = np.sort(i)
    k = 1
    if randomize_polarity and mode=='train_paramloss':
        k = np.random.randint(2, size=(N,1))*2-1
    if randomize_phase and mode=='train_paramloss':
        def rot(x,r):
            return np.hstack([x[r:], x[:r]])
        x = np.array([rot(x, r)
                      for x,r in zip(d[i,1:],
                                     np.random.randint((d.shape[1]-1)//4,
                                                       size=N))])
        return k*x, d[i,y_index]
    else:
        return k*d[i,1:], d[i,y_index]

def sample_params(y):
    r = []
    for v in y:
        i = 0
        while np.sum(i) < 1:
            i = (data[:,0].astype(int)==int(v))
            v = np.max(v+np.random.randint(3)-1, 0)
        r.append(np.argmax(i))
    return data[r,1:], data[r,y_index]

def sample_all(mode='train',noise=True):
    i = {'train':training_indexes,
         'test':testing_indexes,
         None:slice(None)}[mode]
    x,y = data[i, 1:], data[i, y_index]
    nz = 0
    if mode=='train' and noise:
        nz = np.random.uniform(0,0.1,size=x.shape)
    return x+nz, y

sr = 44100.0
param_domain = [[0.0, 4.0]]
domain_size = data.shape[1]-1
data_domain = np.arange(domain_size)/sr

## Functions to put data and parameters in the range [-1,1]

# Data already normalized but too large, divide down
if mode == 'diffed':
    def encode_data(x):
        return (x-means)*10; (x - means) / stds

    # Undo mean/std scaling
    def decode_data(x):
        return x/10+means; x * stds + means
else:
    def encode_data(x):
        return x
    def decode_data(x):
        return x

# Params are from 0 to 4
def encode_params(y):
    return y / 2.0 - 1

# Undo range scaling
def decode_params(y):
    return (y + 1) * 2.0

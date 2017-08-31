
__all__ = ['sample_random', 'sample_params', 'sample_all',
           'data_domain', 'param_domain'
           'encode_data', 'decode_data',
           'encode_params', 'decode_params',
           'domain_size', 'name', 'sample_rate']

import h5py
import numpy as np
from scipy.signal import lfilter, detrend

name = 'bowed3'

#mode = 'rotated'
mode = 'diffed'

sample_rate = 48000

use_half_pos_range = True
randomize_phase = False
randomize_polarity = False

beg = None
beg = 133508
rows = None
rows = 100000

# h = h5py.File('bowed.h5', mode='r')
# data = h['mode']

# Surprisingly slower to have data disk-based, so make a
# memory-resident copy
with h5py.File('bowed3.h5', mode='r') as h:
    data = np.copy(h['rotated'])
    stds = np.copy(h[mode+'_std'][4:])
    means = np.copy(h[mode+'_mean'][4:])

if mode=='diffed':
    data = np.hstack([data[:,:4],
                      data[:,5:] - data[:,4:-1]])
if use_half_pos_range:
    data = np.array([d for d in data[:,:] if d[1] < 64])
if rows is not None:
    if beg is None:
        beg = np.random.randint(0, data.shape[0]-rows, size=1)[0]
    print('beg =',beg)
    data = data[beg:beg+rows,:]

random_indexes = np.arange(data.shape[0])
np.random.shuffle(random_indexes)
testing_indexes = random_indexes[:len(random_indexes)//4]
training_indexes = random_indexes[len(random_indexes)//4:]

y_index = slice(0,2) # pressure, position (ignoring velocity=2, volume=3)

shuffled_indexes = np.array([])
shuffled_pos = 0
epoch_count = 0
def sample_random(N,mode='train'):
    global shuffled_indexes, shuffled_pos, epoch_count
    if N<=0: return np.array([]), np.array([])
    d = data[{'train':training_indexes,
              'train_paramloss':training_indexes,
              'test':testing_indexes,
              None:slice(None)}[mode], :]

    if mode=='train' or mode=='train_paramloss':
        if shuffled_pos+N >= shuffled_indexes.shape[0]:
            epoch_count += 1
            i = np.arange(d.shape[0])
            # if use_half_pos_range:
            #     i = np.array([x for x in i if d[x,1] < 64])
            np.random.shuffle(i)
            shuffled_indexes = i
            shuffled_pos = 0

        # Step through windows of shuffled indexes to cover all the samples
        i = shuffled_indexes[shuffled_pos:shuffled_pos+N]
        shuffled_pos += N
    else:
        i = np.arange(d.shape[0])
        if use_half_pos_range:
            i = np.array([x for x in i if d[x,1] < 64])
        np.random.shuffle(i)
        i = i[:N]

    k = 1
    if randomize_polarity and mode=='train_paramloss':
        k = np.random.randint(2, size=(N,1))*2-1
    if randomize_phase and mode=='train_paramloss':
        def rot(x,r):
            return np.hstack([x[r:], x[:r]])
        x = np.array([rot(x, r)
                      for x,r in zip(d[i,4:],
                                     np.random.randint((d.shape[1]-4)//4,
                                                       size=N))])
        return k*x, d[i,y_index]
    else:
        return k*d[i,4:], d[i,y_index]
    # Note: how to apply std/mean normalization and be able to "undo"
    # it in the presence of rotation operation?  Have to pass 'r' back
    # into decode function.

def sample_params(y,only_one=True):
    r = []
    for p,v in y:
        i = 0
        while np.sum(i) < 1:
            i = np.logical_and((data[:,0]+0.5).astype(int)==int(p+0.5),
                               (data[:,1]+0.5).astype(int)==int(v+0.5))
            p = np.max(p+np.random.randint(3)-1, 0)
            v = np.max(v+np.random.randint(3)-1, 0)
        if only_one:
            r.append(np.argmax(i))
        else:
            r += list(np.where(i)[0])
    return data[r,4:], data[r,y_index]

def sample_all(mode='train',noise=True):
    i = {'train':training_indexes,
         'train_paramloss':training_indexes,
         'test':testing_indexes,
         None:slice(None)}[mode]
    np.random.shuffle(i)
    #i = i[:4000]
    if use_half_pos_range:
        i = data[i, 1] < 64
    x,y = data[i, 4:], data[i, y_index]
    return x, y

sr = 48000.0
param_domain = [[0, 128.0], [0, 64+64*(1-use_half_pos_range)]]
domain_size = data.shape[1]-4
data_domain = np.arange(domain_size)/sr

## Functions to put data and parameters in the range [-1,1]

# Data already normalized but too small
def encode_data(x):
    return (x - means) / stds / 4 # * 4

# Undo mean/std scaling
def decode_data(x):
    if mode=='diffed':
        return detrend(lfilter([1],[1,-1],x=(x * 4 * stds + means))) * 2
    else:
        return x * stds + means

# Params are from 0 to 128
def encode_params(y):
    return ((y - [param_domain[0][0], param_domain[1][0]])
            / [(param_domain[0][1]-param_domain[0][0])/2,
               (param_domain[1][1]-param_domain[1][0])/2] - 1)

# Undo range scaling
def decode_params(y):
    return ((y + 1) * [(param_domain[0][1]-param_domain[0][0])/2,
                       (param_domain[1][1]-param_domain[1][0])/2]
            + [param_domain[0][0], param_domain[1][0]])


__all__ = ['sample_random', 'sample_params', 'sample_all',
           'data_domain', 'param_domain'
           'encode_data', 'decode_data',
           'encode_params', 'decode_params',
           'domain_size', 'name', 'sample_rate']

import h5py
import numpy as np
from scipy.signal import lfilter, detrend

name = 'bowed'
#mode = 'logfft'
mode = 'normalized'
#mode = 'rotated'

sample_rate = 48000

use_half_pos_range = True
randomize_phase = False
randomize_polarity = False

# h = h5py.File('bowed.h5', mode='r')
# data = h['mode']

# Surprisingly slower to have data disk-based, so make a
# memory-resident copy
with h5py.File('bowed.h5', mode='r') as h:
    data = np.copy(h[mode])
    if use_half_pos_range:
        data = np.array([d[:] for d in data if d[1] < 64])
    stds = np.copy(h['std'])
    means = np.copy(h['mean'])

if mode == 'rotated':
    stds = np.std(data[:,4:],axis=0)
    means = np.mean(data[:,4:],axis=0)

random_indexes = np.arange(data.shape[0])
np.random.shuffle(random_indexes)
testing_indexes = random_indexes#[:len(random_indexes)//4]
training_indexes = random_indexes#[len(random_indexes)//4:]

# f = encode_data_fft(data[:,4:])
# m = f.mean(axis=0).reshape(1,-1)
# s = f.max(axis=0).reshape(1,-1) - f.min(axis=0).reshape(1,-1) + 0.00001
# f = np.hstack([data[:,:4], (f - m)/s])

# with h5py.File('bowed.h5', mode='r+') as h:
#     del h['logfft-mean']
#     del h['logfft-std']
#     # h.create_dataset('logfft', data=f, chunks=(f.shape[1],100),
#     #                  compression='gzip', compression_opts=9)
#     h.create_dataset('logfft-mean', data=m)
#     h.create_dataset('logfft-std', data=s)
#     h['logfft'][:,:] = f

if mode=='logfft':
    with h5py.File('bowed.h5', mode='r') as h:
        stds = np.copy(h['logfft-std'])
        means = np.copy(h['logfft-mean'])

y_index = slice(0,2) # pressure, position (ignoring velocity=2, volume=3)

sample_indexes = []
sample_indexes_pos = 0

def sample_random(N,mode=None):#mode='train'):
    global sample_indexes, sample_indexes_pos
    if N<=0: return np.array([]), np.array([])
    i = np.array([],dtype=int)
    d = data[{'train':training_indexes,
              'train_paramloss':training_indexes,
              'test':testing_indexes,
              None:slice(None)}[mode], :]
    if use_half_pos_range:
        d = d[d[:,1] < 64]
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
                      for x,r in zip(d[i,4:],
                                     np.random.randint((d.shape[1]-4)//4,
                                                       size=N))])
        return k*x, d[i,y_index]
    else:
        return k*d[i,4:], d[i,y_index]

def sample_params(y):
    r = []
    for p,v in y:
        i = 0
        while np.sum(i) < 1:
            i = np.logical_and(data[:,0]==int(p), data[:,1]==int(v))
            p = np.max(p+np.random.randint(3)-1, 0)
            v = np.max(v+np.random.randint(3)-1, 0)
        r.append(np.argmax(i))
    return data[r,4:], data[r,y_index]

def sample_all(mode='train',noise=True):
    i = {'train':training_indexes,
         'train_paramloss':training_indexes,
         'test':testing_indexes,
         None:slice(None)}[mode]
    if use_half_pos_range:
        i = [x for x in i if data[x, 1] < 64]
    x,y = data[i, 4:], data[i, y_index]
    return x, y

sr = 48000.0
param_domain = [[0, 128.0], [0, 64+64*(1-use_half_pos_range)]]
domain_size = data.shape[1]-4
data_domain = np.arange(domain_size)/sr

## Functions to put data and parameters in the range [-1,1]

# Data already normalized but too large, divide down
def encode_data(x):
    if mode=='normalized':
        return x / 4
    else:
        return (x - means) / stds / 8

# Undo mean/std scaling
def decode_data(x):
    if mode=='normalized':
        return detrend(lfilter([1],[1,-1],x=x * 4 * stds + means) * 4)
    else:
        return ((x * 8 * stds) + means) * 2

# Encoding by log-FFT (actually log does nothing, it's just the complex values)
def encode_data_fft(x):
    fx = [np.fft.rfft(i) for i in x]
    return ((np.log(np.array([np.hstack([f.real, f.imag])[:-1] for f in fx])+1000)-np.log(1000))/2 - means) / stds

def decode_data_fft(x):
    if len(np.shape(x))==2:
        l = int((np.shape(x)[1]+1)//2)
        y = np.exp((x*4+means)/2+np.log(1000))-1000
        fx = y[:,:l] + np.hstack([y[:,l:],np.zeros((np.shape(x)[0],1))])*1.0j
        return np.array([np.hstack([np.fft.irfft(i), [0]]) for i in fx])
    elif len(np.shape(x))==1:
        l = int((np.shape(x)[0]+1)//2)
        y = (np.exp((x*stds*4+means)/2+np.log(1000))-1000).reshape(-1)
        fx = y[:l] + np.hstack([y[l:],[0]])*1.0j
        return np.hstack([np.fft.irfft(fx), [0]])

if mode=='logfft':
    #already encoded in file
    #encode_data = encode_data_fft
    decode_data = decode_data_fft

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


from matplotlib import pyplot as plt
import numpy as np
import h5py

data = np.loadtxt('bowed.csv', delimiter=',')

# Align maximum correlation with first sample by rotating data around
# the cycle

# A sampling
sample = data[np.random.randint(data.shape[0], size=10),4:]

def rotate(x,n):
    if n==0: return x
    return np.hstack([x[-n:], x[:-n]])

# TODO: Not sure rotation is totally optimal, even correlation seems
# to produce distinct groups of differently-rotated data.  Better when
# restricted to only 40 steps max.

def bestcorr(x):
    sp = np.array([(s - np.mean(s)) / np.linalg.norm(s) for s in sample])
    N = np.shape(x)[0]
    cpos = np.zeros(N)
    cneg = np.zeros(N)
    for i in range(N):
        a = rotate(x,i) - np.mean(x)
        a = a/np.linalg.norm(a)
        cpos[i] = np.sum([np.correlate(a, s)[0] for s in sp])
        cneg[i] = np.sum([np.correlate(-a, s)[0] for s in sp])
    if np.max(cpos) > np.max(cneg):
        return 1,np.argmax(cpos)
    else:
        return -1,np.argmax(cneg)

bc = [bestcorr(x[4:]) for x in data]
rotated = np.array([rotate(x[4:],bc[n][1])*bc[n][0] for n,x in enumerate(data)])

r_concated = np.hstack([data[:,:4], rotated])

diffed = rotated[:,1:] - rotated[:,:-1]
d_concated = np.hstack([data[:,:4], diffed])

means = diffed.mean(axis=0)
stds = diffed.std(axis=0)
normed = (diffed-means)/stds
n_concated = np.hstack([data[:,:4], normed])

with h5py.File('bowed.h5', mode='w') as h:
    cd = lambda n,x: h.create_dataset(n, data=x, compression='gzip',
                                      compression_opts=9,
                                      chunks=(tuple([np.shape(x)[1]]*2)
                                              if len(np.shape(x))==2
                                              else None))
    ds_means = cd('mean', means)
    ds_stds = cd('std', stds)
    ds_freq = cd('frequency', [476.5])
    ds_sr = cd('samplerate', [48000.0])
    ds_rot = cd('rotated', r_concated)
    ds_dif = cd('diffed', d_concated)
    ds_norm = cd('normalized', n_concated)

print('Wrote bowed.h5')

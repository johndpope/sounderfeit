
# load the "known.csv" data and put it in an h5 file

import numpy as np
import h5py
from matplotlib import pyplot as plt

data = np.loadtxt('known.csv', delimiter=',')

# sample bowed data to get alignment of periods
with h5py.File('bowed.h5', mode='r') as f:
    bowed = np.copy(f['rotated'])
    i = np.sort(np.unique(np.random.randint(bowed.shape[0],size=50)))
    samples = np.copy(bowed[i,4:])

d = np.zeros(np.correlate(data[:,2],samples[0]).shape[0])
for s in samples:
    d += np.correlate(data[:,2],s)

peaks = []
last = 0
for i in range(d.shape[0]-200):
    p = np.argmax(d[i:i+200])+i
    if p != last:
        peaks.append((p, d[p]/100))
    last = p
peaks = np.array(peaks)

periods = []
for pos in peaks:
    pos = int(pos[0])
    if pos < 201: continue
    if pos+200 > data.shape[0]: continue
    # find the period in bowed that best corresponds to the parameters
    # and use it to find the phase alignment
    p, v = data[pos-100 : pos+101, :2].mean(axis=0)
    same = bowed[np.logical_and(bowed[:,0]==int(p), bowed[:,1]==int(v)), 4:]
    k = 1
    c = 99
    if same.shape[0] > 0:
        cor = np.convolve(same[0][::-1], data[pos-201 : pos+200, 2])
        if -cor.min() > cor.max():
            c = cor.argmin()
            k = -1
        else:
            c = cor.argmax()
            k = 1
    c = int(c%101)
    periods.append(np.hstack([[p, v], k*data[pos-100+c : pos+101+c, 2]]))

periods = np.array(periods)
assert periods.shape[1]==203
diffed = np.hstack([periods[:,:2], periods[:,3:] - periods[:,2:-1]])
assert diffed.shape[1]==202

print(periods.shape)

with h5py.File('known.h5',mode='w') as h:
    h.create_dataset('periods', data=periods)
    h.create_dataset('diffed', data=diffed)

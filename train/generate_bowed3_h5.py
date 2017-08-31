
from matplotlib import pyplot as plt
import numpy as np
import h5py, os, time

random_sample = []

with open('bowed3.csv', mode='r') as f:
    f.seek(0, os.SEEK_END)
    n = f.tell()
    for i in range(30):
        f.seek(np.random.randint(n-1000))
        f.readline()
        d = f.readline()
        d = np.fromiter(map(float,d.strip().split(',')), dtype=float)
        random_sample.append(d)

def rot(x,r):
    return np.hstack([x[r:], x[:r]])

plt.clf()
N = 965914
#N = 100000
t0 = time.time()
with h5py.File('bowed3_.h5', mode='w') as h:
    dsR = h.create_dataset('rotated', (N, 205), chunks=(1000,205))
    #dsD = h.create_dataset('diffed', (N, 204), chunks=(1000,204))
    dsD = np.zeros((N,204), dtype=float)
    with open('bowed3.csv', mode='r') as f:
        i = 0
        for d in f:
            d = np.fromiter(map(float,d.strip().split(',')), dtype=float)
            params = d[:4]
            d = d[4:]
            # cor = np.zeros(201//2,dtype=float)
            # for s in random_sample:
            #     for r in range(201//2):
            #         c = np.correlate(s[4:], rot(d,r))[0]
            #         cor[r] += np.abs(c)
            # r = rot(d,cor.argmax())
            if d.max() > -d.min():
                r = rot(d,d.argmax())
            else:
                r = rot(d,d.argmin())
            dsR[i,:4] = params
            dsR[i,4:] = r
            dsD[i,:4] = params
            dsD[i,4:] = r[1:] - r[:-1]
            i += 1
            t = time.time()-t0
            print('%0.02f%%, ETA %d min     '%(i * 100.0 / N, (t/i*N-t) / 60.0),
                  end='\r')
            if i >= N:
                break

    m = np.mean(dsR, axis=0)
    s = np.std(dsR, axis=0)
    h.create_dataset('rotated_mean', data = m)
    h.create_dataset('rotated_std', data = s)
    # dsR[:,4:] = ((dsR - m) / s)[:,4:]

    m = np.mean(dsD, axis=0)
    s = np.std(dsD, axis=0)
    h.create_dataset('diffed_mean', data = m)
    h.create_dataset('diffed_std', data = s)
    # dsD[:,4:] = ((dsD - m) / s)[:,4:]

    # print('np.mean(ds, axis=0): ',np.mean(ds, axis=0))
    # print('np.std(ds, axis=0): ',np.std(ds, axis=0))

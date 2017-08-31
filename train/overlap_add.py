
import numpy as np
from scipy.signal import gaussian, hanning

# Note: For 2x overlap use Hann, for 4x overlap use Blackman
# http://www.katjaas.nl/FFTwindow/FFTwindow.html

# Call f(p) to generate a window for each set of params and
# overlap-add them after applying an envelope.
def overlap_add(params, f):
    res = None
    env = []
    for p in params:
        x = f(p)
        hL = len(x)//2
        if len(env) != len(x):
            env = hanning(len(x),False)
        if res is None:
            res = x*env
        else:
            w = x*env
            res = np.hstack([res[:-hL], res[-hL:] + w[:hL], w[hL:]])
    return res

if __name__=="__main__":
    from matplotlib import pyplot as plt
    def test(p):
        return np.ones(100)*p[0]
    x = overlap_add([[1]]*40, test)
    plt.clf()
    plt.plot(x)
    plt.show()

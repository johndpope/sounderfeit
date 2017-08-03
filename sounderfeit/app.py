
from sounderfeit.gui import dialog
from soundersynth import Soundersynth

import os, pickle

def run():
    synth = Soundersynth()
    fn = os.path.join(os.path.dirname(__file__),'..','data','autoenc_decoder.pickle')
    weights = pickle.load(open(fn,'rb')) # w3, w4, b3, b4
    synth.setDecoderWeights(weights)
    print('Input size:',synth.decoderInputSize())
    print('Hidden size:',synth.decoderHiddenSize())
    print('Output size:',synth.decoderOutputSize())
    synth.decodeCycle();
    #print(weights[0])
    return dialog.dialog(synth)

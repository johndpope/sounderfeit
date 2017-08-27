
from sounderfeit.gui import dialog
from soundersynth import Soundersynth

import os, pickle

decoders = [('bowed-D1Z2Y',   'bowed-half-D1Z2Y-decoder.pickle'),
            ('bowed2-D1Z2Y',  'bowed3-half-D1Z2Y-decoder.pickle'),
            ('vowelsD1Z1Y', 'vowels-D1Z1Y-decoder.pickle'),
            ('vowelsD1Z0Y', 'vowels-D1Z0Y-decoder.pickle'),
            ('vowelsN1Z0Y', 'vowels-N1Z0Y-decoder.pickle')]

def run():
    synth = Soundersynth()

    def set_decoder(name):
        filename = dict(decoders)[name]
        fn = os.path.join(os.path.dirname(__file__),'..','data',filename)
        print(fn)
        weights = pickle.load(open(fn,'rb')) # w3, w4, b3, b4
        synth.setDecoderWeights(weights)
        print('Input size:',synth.decoderInputSize())
        print('Hidden size:',synth.decoderHiddenSize())
        print('Output size:',synth.decoderOutputSize())
        synth.decodeCycle()
        #print(weights[0])

    set_decoder(decoders[0][0])
    return dialog.dialog([d[0] for d in decoders], set_decoder, synth)

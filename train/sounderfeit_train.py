##!/usr/bin/env python3

### Adversarial Autoencoder

show_plot = True
show_titles = True
if not show_plot:
    import matplotlib
    matplotlib.use('Agg')
verbose = True
plot_results = True

import os # silence tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import lfilter, gaussian
from scipy.stats import linregress
import sys, json, pickle
import h5py

if not show_titles:
    plt.suptitle = lambda x: None
figdim = 8

# Remove subplot axis frames
def subplot(i,j=None,k=None,left=None):
    args = [i] + [[j],[]][j is None] + [[k],[]][k is None]
    spl = plt.subplot(*args)
    spl.axes.spines['right'].set_visible(False)
    spl.axes.spines['top'].set_visible(False)
    spl.axes.yaxis.set_ticks_position('left')
    spl.axes.xaxis.set_ticks_position('bottom')
    if left is False:
        spl.axes.spines['left'].set_visible(False)
    return spl

import overlap_add as synth

### Data

# import dataset_gaussian_window as ds
# n_data = 200
# ds.domain_size = n_data

import bowed as ds
#import bowed3 as ds
#import vowels as ds

n_data = ds.domain_size

import importlib; importlib.reload(ds)
plt.ion()

### Network parameters
n_z = 1
n_en_hidden = 100
n_de_hidden = 100
n_di_hidden = 100
n_y = len(ds.param_domain)-1
z_dist = lambda size: np.random.normal(0, 0.6827, size=size)
#z_dist = lambda size: np.random.uniform(-1.0,1.0, size=size)

### Training parameters
batch_size = 50
n_batches = 4000

# choices
denoising_method =  ['normal', 'salt'][0]
autoencoder_trainer = ['sgdm', 'adam'][0]
params_tanh_limit = False
fix_phase_for_known = True

# params
denoising_std   = 0.0 # std normal (if selected)
denoising_prob  = 0.5 # probability zero (if selected)
instance_noise  = 0.0 # std normal (TODO: should be annealed)
coefE_paramloss  = 0.5 # coefficient for parameter loss (lambda_encoder)
#coefD_paramloss  = 0.1 # coefficient for parameter loss (lambda_decoder)
learning_rate_ae = 0.005
momentum_ae = 0.001
learning_rate_disc = 0.05
momentum_disc = 0.001

enable_pretraining = False

# Set to true to enable the autoencoder (otherwise CGAN)
enable_autoencoder = True

# Set to true to expose data to the discriminator
# Observation: generates ok curves by itself (y-disc only) but
# severely distorts things when autoencoder is enabled.
disc_data = False

# Set to true to enable parameter losses
enable_param_loss = True

# Set to true to enable backprob on Dy
# Observation: Screws up z-disc badly
enable_param_loss_extra = False

# Set to true to enable the adversarial regularization
enable_z_discriminator = True

# Note: y-disc seems almost always useless?
enable_y_discriminator = False

# Set to true to enable data-parameter discriminator
# Observation: Totally biased and weird parameter distribution, tends
# to generate the same curve.  In the latent parameter, create X
# patterns, i.e. z,y are highly positively or negatively correlated.
# Trying now to expose it to z as well as y, to see if that removes
# that weirdness.
# Yes it did but trying again with z-disc and y-disc disabled.
# Finally: Yes, better, seemed to perform the combined function of z-
# and y-disc, however it diverged. Unclear whether it's
# better/comparable.
# If parameter loss is turned off, y0 piles up around 0.3, very weird
# distribution that doesn't match the prior, however z0 distribution
# is normal-looking around 0.
enable_dpdisc = False

# A prefix to add to all saved files for identifying configuration
if len(sys.argv)>1 and 'ipython' not in sys.argv[0]:
    prefix=sys.argv[1]
else:
    prefix='wip'

# cmdline stuff
if len(sys.argv)>2 and 'ipython' not in sys.argv[0]:
    args=json.loads(sys.argv[2])
    for a,v in args.items():
        globals()[a] = v
# save config
os.system('mkdir -p results')
with open('results/'+prefix+'_config.txt','w') as f:
    g = {a:v for a,v in globals().items() if isinstance(v,int) or isinstance(v,float) or isinstance(v,str)}
    print(json.dumps(g,indent=True,skipkeys=True), file=f)

### Weights
tf.reset_default_graph()
with tf.variable_scope('NN'):
    # Encoder
    w1 = tf.get_variable(name='w1', shape=[n_data, n_en_hidden],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b1 = tf.get_variable(name='b1', shape=[n_en_hidden],
                         initializer=tf.zeros_initializer())
    w2 = tf.get_variable(name='w2', shape=[n_en_hidden, n_z + n_y],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b2 = tf.get_variable(name='b2', shape=[n_z + n_y],
                         initializer=tf.zeros_initializer())

    # Decoder
    w3 = tf.get_variable(name='w3', shape=[n_z + n_y, n_de_hidden],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b3 = tf.get_variable(name='b3', shape=[n_de_hidden],
                         initializer=tf.zeros_initializer())
    w4 = tf.get_variable(name='w4', shape=[n_de_hidden, n_data],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b4 = tf.get_variable(name='b4', shape=[n_data],
                         initializer=tf.zeros_initializer())

    # Discriminator: real or fake?
    w5 = tf.get_variable(name='w5', shape=[n_z, n_di_hidden],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b5 = tf.get_variable(name='b5', shape=[n_di_hidden],
                         initializer=tf.zeros_initializer())
    w6 = tf.get_variable(name='w6', shape=[n_di_hidden, 1],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b6 = tf.get_variable(name='b6', shape=[1],
                         initializer=tf.zeros_initializer())

    # Categorical Discriminator: real or fake?
    w9 = tf.get_variable(name='w9', shape=[n_data*disc_data + n_y*(not disc_data), n_di_hidden],
                         initializer=tf.random_uniform_initializer(-0.01, 0.01))
    b9 = tf.get_variable(name='b9', shape=[n_di_hidden],
                         initializer=tf.zeros_initializer())
    w10 = tf.get_variable(name='w10', shape=[n_di_hidden, 1],
                          initializer=tf.random_uniform_initializer(-0.01, 0.01))
    b10 = tf.get_variable(name='b10', shape=[1],
                          initializer=tf.zeros_initializer())

    # Data-Param Discriminator: do X and Y go together?
    w7 = tf.get_variable(name='w7', shape=[n_data + n_z*0 + n_y*0, n_di_hidden],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b7 = tf.get_variable(name='b7', shape=[n_di_hidden],
                         initializer=tf.zeros_initializer())
    w8 = tf.get_variable(name='w8', shape=[n_di_hidden, 1],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b8 = tf.get_variable(name='b8', shape=[n_y],
                         initializer=tf.zeros_initializer())

    w71 = tf.get_variable(name='w71', shape=[n_y + n_y, n_y],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b71 = tf.get_variable(name='b71', shape=[n_y],
                         initializer=tf.zeros_initializer())
    w81 = tf.get_variable(name='w81', shape=[n_y, 1],
                         initializer=tf.random_uniform_initializer(-0.1, 0.1))
    b81 = tf.get_variable(name='b81', shape=[1],
                         initializer=tf.zeros_initializer())

### Network
z = tf.placeholder(tf.float32, shape=[None, n_z], name="z")    # Latent variables
y = tf.placeholder(tf.float32, shape=[None, n_y], name="y")    # Parameters
x = tf.placeholder(tf.float32, shape=[None, n_data], name="x") # Data to reconstruct
x_in = tf.placeholder(tf.float32, shape=[None, n_data], name="x") # Data in

zy = tf.concat(axis=1, values=[z,y])

act = tf.nn.relu
prob = tf.nn.sigmoid

# noisy x
if denoising_method == 'normal':
    nx = tf.random_normal(shape=tf.shape(x), mean=0.0,
                          stddev=denoising_std, dtype=tf.float32) + x_in

# salt & pepper noisy x
elif denoising_method == 'salt':
    nx = tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0.0,
                                   maxval=1.0, dtype=tf.float32)>denoising_prob,
                 tf.float32) * x_in

# no noise, but take potentially-corrupted input
else:
    nx = x_in

# Encoder: Generate latent z based on data
if params_tanh_limit:
    E = tf.nn.tanh( tf.matmul( act(tf.matmul(x, w1) + b1), w2 ) + b2 )
    nE = tf.nn.tanh( tf.matmul( act(tf.matmul(nx, w1) + b1), w2 ) + b2 )
else:
    E = ( tf.matmul( act(tf.matmul(x, w1) + b1), w2 ) + b2 )
    nE = ( tf.matmul( act(tf.matmul(nx, w1) + b1), w2 ) + b2 )

Ez = E[:,:n_z]
nEz = nE[:,:n_z]

# Decoder: Generate data x based on latent z + y
nEy  = tf.concat(axis=1, values=[nEz,y])
Ey  = tf.concat(axis=1, values=[Ez,y])
nDey = (tf.matmul( act(tf.matmul(nE, w3) + b3), w4 ) + b4)
Dey = (tf.matmul( act(tf.matmul(E, w3) + b3), w4 ) + b4)
Dzy = (tf.matmul( act(tf.matmul(zy, w3) + b3), w4 ) + b4)
nDy  = (tf.matmul( act(tf.matmul(nEy, w3) + b3), w4 ) + b4)
Dy  = (tf.matmul( act(tf.matmul(Ey, w3) + b3), w4 ) + b4)
# Dey = tf.matmul( act(tf.matmul(E - b2, -tf.transpose(w2))) - b1, -tf.transpose(w1) )
# Dzy = tf.matmul( act(tf.matmul(zy - b2, -tf.transpose(w2))) - b1, -tf.transpose(w1) )
# Dy = tf.matmul( act(tf.matmul(Ey - b2, -tf.transpose(w2))) - b1, -tf.transpose(w1) )

# Discriminator: Determine if data is real or generated
# if disc_data:
#     # Dg = prob( tf.matmul( act(tf.matmul(DE, w5) + b5), w6 ) + b6 )
#     # Dx = prob( tf.matmul( act(tf.matmul(xzy, w5) + b5), w6 ) + b6 )
#     Dg = prob( tf.matmul( act(tf.matmul(Dey, w5) + b5), w6 ) + b6 )
#     Dx = prob( tf.matmul( act(tf.matmul(x, w5) + b5), w6 ) + b6 )
# else:
#     # Dg = prob( tf.matmul( act(tf.matmul(E, w5) + b5), w6 ) + b6 )
#     # Dx = prob( tf.matmul( act(tf.matmul(zy, w5) + b5), w6 ) + b6 )

# noisy z/Ez (instance noise, input to discriminator)
iEz = tf.random_normal(shape=tf.shape(nEz), mean=0.0,
                       stddev=instance_noise, dtype=tf.float32) + nEz
iz = tf.random_normal(shape=tf.shape(z), mean=0.0,
                      stddev=instance_noise, dtype=tf.float32) + z

Dg = prob( tf.matmul( act(tf.matmul(Ez, w5) + b5), w6 ) + b6 )
Dx = prob( tf.matmul( act(tf.matmul(z, w5) + b5), w6 ) + b6 )

# Categorical discriminator
DEy = tf.concat(axis=1, values=[nDey,E[:,n_z:n_z+n_y]])
xy = tf.concat(axis=1, values=[x,y])
if disc_data:
    # CatDg = prob( tf.matmul( act(tf.matmul(DEy, w9) + b9), w10 ) + b10 )
    # CatDx = prob( tf.matmul( act(tf.matmul(xy, w9) + b9), w10 ) + b10 )
    CatDg = prob( tf.matmul( act(tf.matmul(nDey, w9) + b9), w10 ) + b10 )
    CatDx = prob( tf.matmul( act(tf.matmul(x, w9) + b9), w10 ) + b10 )

    # Above didn't work for CatDg, so try only training the decoder
    # CatDg = prob( tf.matmul( act(tf.matmul(Dzy, w9) + b9), w10 ) + b10 )
else:
    CatDg = prob( tf.matmul( act(tf.matmul(E[:,n_z:n_z+n_y], w9) + b9), w10 ) + b10 )
    CatDx = prob( tf.matmul( act(tf.matmul(y, w9) + b9), w10 ) + b10 )

Dy_loss = tf.reduce_mean(tf.square( nDy - x )) # y->x reconstruction

# Autoencoder loss: reconstruction error
if n_y>0:
    Ey_loss = tf.reduce_mean(tf.square( nE[:,n_z:] - y ))  # Parameter reconstruction
else:
    Ey_loss = tf.constant(0.0);
E_loss = (tf.reduce_mean(tf.square( nDey - x ))         # Data reconstruction
          + Ey_loss*coefE_paramloss*enable_param_loss*enable_autoencoder
          )#+ Dy_loss*coefD_paramloss*enable_param_loss*enable_autoencoder)

# E_loss = tf.reduce_sum(tf.nn.softmax([tf.reduce_mean(tf.square( nDey - x )),
#                                       Ey_loss])

# Discrimator loss
#tflog = lambda x: tf.log(tf.maximum(x,1e-9))
tflog = lambda i: tf.log(i + 1e-9)
G_loss = -tf.reduce_mean(tflog(Dg))

# BGAN
# G_loss = tf.reduce_mean(tf.square(tflog(Dg) - tflog(1.0-Dg)))

D_loss = -0.5*tf.reduce_mean(tflog(Dx) + tflog(1. - Dg))

# Categorical discriminator loss
CatDg_loss = -tf.reduce_mean(tflog(CatDg))
# CatDg_loss = tf.reduce_mean(tf.square(tflog(CatDg)-tflog(1.0-CatDg)))
CatDx_loss = -0.5*tf.reduce_mean(tflog(CatDx) + tflog(1. - CatDg))

# DP-Discriminator: Determine if data and parameters go together
xy = tf.concat(axis=1, values=[x,y])
nDy = tf.concat(axis=1, values=[nDey,y])
xEy = tf.concat(axis=1, values=[x,nE[:,n_z:n_z+n_y]])
#xEy = tf.concat(axis=1, values=[x,E])

def rot(x,r):
    return np.hstack([x[r:], x[:r]])

from scipy.signal import detrend
def randrot_feed(feed):
    x_recon = feed[x]
    feed[x_in] = x_recon
    if False:
        feed[x_in] = x_recon
        return feed
    if False:
        ind = np.random.randint(x_recon.shape[1], size=x_recon.shape[0])
        pol = np.random.randint(2, size=(x_recon.shape[0],1))*2-1
        x_input = np.array([rot(x,i) for x,i in zip(x_recon, ind)])
        feed[x_in] = x_input*pol
    if False:
        f = detrend(np.log(np.abs(np.fft.rfft(x_recon)))/10.0)
        feed[x_in] = np.hstack([x_recon[:,:100], f[:,:100]])
    return feed

# TODO trying below
# DPEn = prob( tf.matmul( act(tf.matmul(xEy, w7) + b7), w8 ) + b8 ) #mismatch
# DPDe = prob( tf.matmul( act(tf.matmul(Dy, w7) + b7), w8 ) + b8 )  #mismatch
# DPDi = prob( tf.matmul( act(tf.matmul(xy, w7) + b7), w8 ) + b8 )  #match

# TODO: Try reducing data to same size as parameter vector before
# "comparing" them
DPDe0 = act( tf.matmul( act(tf.matmul(nDey, w7) + b7), w8 ) + b8 )  #mismatch
DPDi0 = act( tf.matmul( act(tf.matmul(x, w7) + b7), w8 ) + b8 )  #match

DPDi0Ey = tf.concat(axis=1, values=[DPDi0,Ey[:,n_z:n_z+n_y]])
DPDe0Dy = tf.concat(axis=1, values=[DPDe0,y])
DPDi0y = tf.concat(axis=1, values=[DPDi0,y])
DPEn = prob( tf.matmul( act(tf.matmul(DPDi0Ey, w71) + b71), w81 ) + b81 ) #mismatch
DPDe = prob( tf.matmul( act(tf.matmul(DPDe0Dy, w71) + b71), w81 ) + b81 ) #mismatch
DPDi = prob( tf.matmul( act(tf.matmul(DPDi0y, w71) + b71), w81 ) + b81 )  #match

# TODO why is the value of DPDi_loss not changing?
DPEn_loss = -tf.reduce_mean(tflog(DPEn))
DPDe_loss = -tf.reduce_mean(tflog(DPDe))
DPEnDe_loss = -tf.reduce_mean(tflog(DPEn) + tflog(DPDe))
# DPEnDe_loss = tf.reduce_mean((tf.square(tflog(DPEn) - tflog(1.-DPEn))
#                               +tf.square(tflog(DPDe) - tflog(1.-DPDe)))/2)

DPDi_loss = -tf.reduce_mean(tflog(DPDi) + (tflog(1. - DPEn) + tflog(1. - DPDe))/2
                            )/2

### Trainers
Ept_train = tf.train.AdamOptimizer().minimize(E_loss) #pre-trainer

# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.01
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            1000, 0.98, staircase=True)

if autoencoder_trainer == 'adam':
    opt = lambda: tf.train.MomentumOptimizer(learning_rate=learning_rate_ae,
                                             momentum=momentum_ae)
elif autoencoder_trainer == 'sgdm':
    opt = lambda: tf.train.AdamOptimizer(learning_rate=learning_rate_ae)
E_train = opt().minimize(E_loss,# + G_loss*0.1*enable_z_discriminator,
                         var_list=[w1,w2,b1,b2,w3,w4,b3,b4]
)#, global_step=global_step)
Dy_train = opt().minimize(Dy_loss)
if n_y > 0:
    Ey_train = opt().minimize(Ey_loss)
opt = lambda: tf.train.MomentumOptimizer(learning_rate=learning_rate_disc,
                                         momentum=momentum_disc)
G_train = opt().minimize(G_loss, var_list=[w1,w2,b1,b2])
D_train = opt().minimize(D_loss, var_list=[w5,w6,b5,b6])
CatDg_train = opt().minimize(CatDg_loss, var_list=[w1,w2,b1,b2,
                                                   w3,w4,b3,b4])
CatDx_train = opt().minimize(CatDx_loss, var_list=[w9,w10,b9,b10])

if enable_dpdisc:
    DPDi_train = opt().minimize(DPDi_loss, var_list=[w7,w8,w71,w81,
                                                     b7,b8,b71,b81])
    DPEnDe_train = opt().minimize(DPEnDe_loss, var_list=[w1,w2,w3,w4,
                                                         b1,b2,b3,b4])
    DPEn_train = opt().minimize(DPEnDe_loss, var_list=[w1,w2,b1,b2])
    DPDe_train = opt().minimize(DPEnDe_loss, var_list=[w3,w4,b3,b4])

### Evaluation function
def evaluate(mode, save=True):
    E_x, E_y = ds.sample_random(min([10000,ds.data.shape[0]]),mode=mode)
    E_z = np.zeros((E_x.shape[0],n_z))
    feed = { z: E_z, x: ds.encode_data(E_x),
             y: ds.encode_params(E_y)[:,:n_y] }
    E_result, D_result, zy_result = s.run([Dey, Dy, E], feed_dict=feed)

    rms_autoencoder = np.sqrt(np.mean((E_result - ds.encode_data(E_x))**2, axis=1))
    rms_decoder = np.sqrt(np.mean((D_result - ds.encode_data(E_x))**2, axis=1))
    if n_y>0:
        rms_encoder = np.sqrt(np.mean((zy_result[:,n_z:]
                                       - ds.encode_params(E_y))**2, axis=1))
    else:
        rms_encoder = [0]
    rms_ae_decoder = np.sqrt(np.mean((D_result - E_result)**2, axis=1))

    rms_autoencoder = np.mean(rms_autoencoder)
    rms_decoder = np.mean(rms_decoder)
    rms_encoder = np.mean(rms_encoder)
    rms_ae_decoder = np.mean(rms_ae_decoder)

    if not save: return [rms_autoencoder, rms_decoder, rms_encoder, rms_ae_decoder]
    if verbose:
        print('Error summary for %s (%s);'%(mode,prefix))
        print('  encoder             rms: %0.04f'%rms_encoder)
        print('  autoencoder         rms: %0.04f'%rms_autoencoder)
        print('  decoder             rms: %0.04f'%rms_decoder)
        print('  autoencoder-decoder rms: %0.04f'%rms_ae_decoder)
    with open('results/'+prefix+'-rms.txt','a') as f:
        f.write(json.dumps({'label': prefix,
                            'mode': mode,
                            'encoder': float(rms_encoder),
                            'autoencoder': float(rms_autoencoder),
                            'decoder': float(rms_decoder),
                            'ae_decoder': float(rms_ae_decoder)},
                           indent=True))
    return [rms_autoencoder, rms_decoder, rms_encoder, rms_ae_decoder]

### Run
trace = []
s = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()

# def filewrite():
#     E_z = np.random.normal(0, 1, size=(5,n_z))
#     E_y = np.array([[100]*5,
#                     np.linspace(ds.param_domain[0][0],
#                                 ds.param_domain[0][1], 7)[1:-1]
#                     ]).T
#     E_x = ds.sample_params(E_y)
#     feed = { z: E_z, x: ds.encode_data(E_x), y: ds.encode_params(E_y) }
#     E_result, D_result = s.run([Dey, Dzy], feed_dict=feed)
#     f = open('test.csv',mode='a')
#     print(','.join(map(str,np.hstack([E_x.reshape(-1,), E_result.reshape(-1,), D_result.reshape(-1,)]))), file=f)

# Pre-training
s.run(tf.global_variables_initializer())
for i in range(10000//batch_size):
    if not enable_pretraining: break
    # Pre-train autoencoder using Adam
    E_z = z_dist(size=(batch_size,n_z))
    E_x, E_y = ds.sample_random(batch_size)
    feed = { z: E_z, x: ds.encode_data(E_x), y: ds.encode_params(E_y) }
    if enable_autoencoder:
        s.run([Ept_train], feed_dict=feed)

if verbose:
    print('Doing {} batches.. ({})'.format(n_batches, prefix))
try:
    for i in range(n_batches):
        E_result=[0,0,0,0]; Dy_result=[0,0]; G_result=[0,0]; D_result=[0,0]
        DPEnDe_result=[0,0]; DPEn_result=[0,0]; DPDe_result=[0,0]; DPDi_result=[0,0]
        CatDg_result=[0,0]; CatDx_result=[0,0]

        # Train autoencoder + generator
        E_z = z_dist(size=(batch_size,n_z))
        E_x, E_y = ds.sample_random(batch_size, mode='train')
        feed = { x: ds.encode_data(E_x), z: E_z,
                 y: ds.encode_params(E_y)[:,:n_y] }
        randrot_feed(feed)
        if enable_autoencoder:
            E_result = s.run([E_train, E_loss, Ey_loss, Dy_loss], feed_dict=feed)
        elif enable_param_loss and n_y > 0 and not ds.randomize_phase:
            # We train on Ey regardless, otherwise there is no
            # conditioning and it's not a CGAN.
            E_result = s.run([Ey_train, E_loss, Ey_loss, Dy_loss], feed_dict=feed)
        else:
            E_result = [0] + list(s.run([E_loss, Ey_loss, Dy_loss], feed_dict=feed))

        # Train on Ey separately so that we can introduce random phase
        # rotation during parameter training but not during
        # autoencoder training
        if enable_param_loss and n_y > 0 and ds.randomize_phase:
            E_x, E_y = ds.sample_random(batch_size, mode='train_paramloss')
            feed = { x: ds.encode_data(E_x),
                     y: ds.encode_params(E_y)[:,:n_y] }
            randrot_feed(feed)
            E_result = s.run([Ey_train, E_loss, Ey_loss, Dy_loss], feed_dict=feed)

        # Train (enhance) decoder parameter-only mapping
        if enable_param_loss_extra:
            D_z = z_dist(size=(batch_size,n_z))
            D_x, D_y = ds.sample_random(batch_size)
            feed = { z: D_z, y: ds.encode_params(D_y)[:,:n_y],
                     x: ds.encode_data(D_x) }
            randrot_feed(feed)
            Dy_result = s.run([Dy_train, Dy_loss], feed_dict=feed)

        # if (i%4)!=0:
        #     continue

        if enable_z_discriminator:
            D_z = z_dist(size=(batch_size,n_z))
            D_x, D_y = ds.sample_random(batch_size)
            feed = { z: D_z, x: ds.encode_data(D_x) }
            randrot_feed(feed)
            D_result = s.run([D_train, D_loss], feed_dict=feed)

        # Train encoder by discriminator (generator)
        if enable_z_discriminator:
            G_x, G_y = ds.sample_random(batch_size)
            feed = { x: ds.encode_data(G_x) }
            randrot_feed(feed)
            G_result = s.run([G_train, G_loss], feed_dict=feed)
            #G_result = [0, s.run([G_loss], feed_dict=feed)[0]]

        # Train encoder by categorical discriminator
        if enable_y_discriminator:
            D_z = z_dist(size=(batch_size,n_z))
            if disc_data:
                D_x, _ = ds.sample_random(batch_size)
                _, D_y = ds.sample_random(batch_size)
            else:
                D_x, D_y = ds.sample_random(batch_size)
            feed = { z: D_z, x: ds.encode_data(D_x),
                     y: ds.encode_params(D_y)[:,:n_y] }
            randrot_feed(feed)
            CatDx_result = s.run([CatDx_train, CatDx_loss], feed_dict=feed)

        # Train encoder by categorical discriminator (generator)
        if enable_y_discriminator:
            G_z = z_dist(size=(batch_size,n_z))
            if disc_data:
                G_x, _ = ds.sample_random(batch_size)
                _, G_y = ds.sample_random(batch_size)
            else:
                G_x, G_y = ds.sample_random(batch_size)
            feed = { z: G_z, x: ds.encode_data(G_x),
                     y: ds.encode_params(G_y)[:,:n_y] }
            randrot_feed(feed)
            CatDg_result = s.run([CatDg_train, CatDg_loss], feed_dict=feed)

        if enable_dpdisc:
            E_z = z_dist(size=(batch_size,n_z))
            E_x, E_y = ds.sample_random(batch_size)
            feed = { z: E_z, x: ds.encode_data(E_x),
                     y: ds.encode_params(E_y)[:,:n_y] }
            randrot_feed(feed)
            DPEn_result = s.run([DPEn_train, DPEn_loss], feed_dict=feed)

            E_z = z_dist(size=(batch_size,n_z))
            E_x, E_y = ds.sample_random(batch_size)
            feed = { z: E_z, x: ds.encode_data(E_x),
                     y: ds.encode_params(E_y)[:,:n_y] }
            randrot_feed(feed)
            DPDe_result = s.run([DPDe_train, DPDe_loss], feed_dict=feed)

            E_z = z_dist(size=(batch_size,n_z))
            E_x, E_y = ds.sample_random(batch_size)
            feed = { z: E_z, x: ds.encode_data(E_x),
                     y: ds.encode_params(E_y)[:,:n_y] }
            randrot_feed(feed)
            DPDi_result = s.run([DPDi_train, DPDi_loss], feed_dict=feed)

        if (i % 10)==0:
            #saver.save(s, './save/autogan-'+prefix)
            #break
            #filewrite()
            # print('AE weights:', [np.linalg.norm(i)
            #                       for i in s.run([w1,w2,w3,w4])])
            # print('Disc weights:', [np.linalg.norm(i)
            #                         for i in s.run([w9,w10])])
            trace.append([E_result[1], E_result[2], E_result[3], G_result[1],
                          D_result[1], CatDg_result[1], CatDx_result[1],
                          DPDi_result[1], DPEn_result[1], DPDe_result[1]]
                         + [0,0,0,0,0,0,0,0])
                         # + evaluate('train', save=False)
                         # + evaluate('test', save=False))
            if verbose:
                print(i,#'learning_rate',s.run([learning_rate])[0],
                      'E_loss',trace[-1][0],
                      'Ey_loss',trace[-1][1],
                      'Dy_loss',trace[-1][2],
                      'G_loss',np.exp(-trace[-1][3]),
                      'D_loss',np.exp(-trace[-1][4]))
                      # 'CatDg_loss',np.exp(-trace[-1][5]),
                      # 'CatDx_loss',np.exp(-trace[-1][6]),
                      # 'DPDi_loss',np.exp(-trace[-1][7]),
                      # 'DPEn_loss',np.exp(-trace[-1][8]),
                      # 'DPDe_loss',np.exp(-trace[-1][9]))
except KeyboardInterrupt:
    pass

# saver = tf.train.import_meta_graph('./save/autogan-'+prefix+'.meta')
# saver.restore(s, tf.train.latest_checkpoint('./save'))
# #s.run(tf.global_variables_initializer())
# graph = tf.get_default_graph()
# Dey = graph.get_tensor_by_name(Dey.name)
# Dzy = graph.get_tensor_by_name(Dzy.name)

# Plot error traces
trace = np.array(trace)
np.savetxt('results/'+prefix+'-error-trace.csv',trace, delimiter=',')
plt.figure(1,figsize=(figdim,figdim)).clear()
plt.suptitle('Loss traces')
n = 5+enable_z_discriminator+enable_y_discriminator+enable_dpdisc
j = 1
subplot(n,1,j)
j += 1
plt.plot(trace[:,0], label='E_loss')
plt.plot(trace[:,1], label='Ey_loss')
plt.plot(trace[:,2], label='Dy_loss')
plt.legend()
subplot(n,1,j)
j += 1
plt.plot(trace[:,10], label='train rms_autoencoder')
plt.plot(trace[:,14], label='test rms_autoencoder')
plt.legend()
subplot(n,1,j)
j += 1
plt.plot(trace[:,11], label='train rms_decoder')
plt.plot(trace[:,15], label='test rms_decoder')
plt.legend()
subplot(n,1,j)
j += 1
plt.plot(trace[:,12], label='train rms_encoder')
plt.plot(trace[:,16], label='test rms_encoder')
plt.legend()
subplot(n,1,j)
j += 1
plt.plot(trace[:,13], label='train rms_ae_decoder')
plt.plot(trace[:,17], label='test rms_ae_decoder')
plt.legend()
if enable_z_discriminator:
    subplot(n,1,j)
    j += 1
    plt.plot(np.exp(-trace[:,3]), label='G_loss')
    plt.plot(np.exp(-trace[:,4]), label='D_loss')
    plt.legend()
if enable_y_discriminator:
    subplot(n,1,j)
    j += 1
    plt.plot(np.exp(-trace[:,5]), label='CatDg_loss')
    plt.plot(np.exp(-trace[:,6]), label='CatDx_loss')
    plt.legend()
if enable_dpdisc:
    subplot(n,1,j)
    j += 1
    #plt.plot(trace[:,7], label='DPEnDe_loss')
    plt.plot(np.exp(-trace[:,7]), label='DPDi_loss')
    plt.plot(np.exp(-trace[:,8]), label='DPEn_loss')
    plt.plot(np.exp(-trace[:,9]), label='DPDe_loss')
    plt.legend()
plt.savefig('results/'+prefix+'-error-trace.png')
plt.savefig('results/'+prefix+'-error-trace.pdf')

# Evaluate performance on full testing and training datasets
with open('results/'+prefix+'-rms.txt','w') as f:  # truncate file
    f.write('[')
evaluate('train', save=True)
with open('results/'+prefix+'-rms.txt','a') as f:
    f.write(',')
evaluate('test', save=True)
with open('results/'+prefix+'-rms.txt','a') as f:
    f.write(']')

# Visualize
if not plot_results:
    import gc; gc.collect()
    exit()

# Save decoder weights
encoder_weights = s.run([w1,w2,b1,b2])
pickle.dump(encoder_weights, open('results/'+prefix+'-encoder.pickle','wb'))
decoder_weights = s.run([w3,w4,b3,b4])
pickle.dump(decoder_weights, open('results/'+prefix+'-decoder.pickle','wb'))

# Evaluate "gaussian" dataset
if ds.name=='gaussian_window':
    E_z = np.random.uniform(-1, 1, size=(7,n_z))
    E_y = np.array([np.linspace(ds.param_domain[0][0],
                                ds.param_domain[0][1], 7),
                    [0.1]*7]).T
    E_x, _ = ds.sample_params(E_y)
    feed = { z: E_z, x: ds.encode_data(E_x), y: ds.encode_params(E_y) }
    E_result, D_result = s.run([Dey, Dzy], feed_dict=feed)

    plt.figure(2)
    for i in range(5):
        subplot(2,5,6+i)
        plt.plot(ds.data_domain, E_x[i])
        plt.plot(ds.data_domain, ds.decode_data(E_result[i]))
        plt.plot(ds.data_domain, ds.decode_data(D_result[i]))

if ds.name=='bowed' or ds.name=='bowed3':
    enc = lambda x: ds.encode_data(x)
    dec = lambda x: ds.decode_data(x)
    # dec = lambda x: x

    # A matrix of both parameters
    r = ds.param_domain[0][1] - ds.param_domain[0][0]
    p = np.linspace(ds.param_domain[0][0]+r*0.1,
                    ds.param_domain[0][1]-r*0.1, 5)
    r = ds.param_domain[1][1] - ds.param_domain[1][0]
    v = np.linspace(ds.param_domain[1][0]+r*0.1,
                    ds.param_domain[1][1]-r*0.1, 5)

    E_z = np.zeros((len(p)*len(v),n_z))
    if n_y==1:
        E_z[:,0] = np.array([np.linspace(-0.8,0.8,5)]*5).reshape(-1)
    if n_y==0:
        E_z[:,0] = np.array([np.linspace(-0.8,0.8,5)]*5).reshape(-1)
        E_z[:,1] = np.array([[x]*5 for x in E_z[:5,0]]).reshape(-1)
    E_y = np.array([(x,y) for x in p for y in v])
    E_x, _ = ds.sample_params(E_y)

    # Find samples closest to z=0
    if ds.name == 'bowed3':
        how_far = []
        if n_y==2:
            n = 0
            for i in range(len(p)):
                for j in range(len(v)):
                    E_x_tmp, E_y_tmp = ds.sample_params([[p[i], v[j]]], only_one=False)
                    # zy_result = s.run([E], feed_dict={x: ds.encode_data(E_x_tmp)})[0]
                    # k = np.mean(
                    #     (np.hstack([zy_result[:, :n_z], ds.encode_params(E_y_tmp)])
                    #      - np.hstack([[0], ds.encode_params(np.array([p[i],v[i]]))])
                    # )**2, axis=1)
                    # if n_z==1: how_far.append(zy_result[k.argmin(), 0])
                    k = np.mean((E_x_tmp - np.mean(E_x_tmp, axis=0))**2, axis=1)
                    E_x[n, :] = E_x_tmp[k.argmin()]
                    # zy_result = s.run([E], feed_dict={x: ds.encode_data(E_x_tmp)})[0]
                    # how_far.append(zy_result[k.argmin(),0])
                    n += 1

    feed = { z: E_z, x: ds.encode_data(E_x),
             y: ds.encode_params(E_y)[:,:n_y] }
    randrot_feed(feed)
    E_result, D_result, zy_result = s.run([Dey, Dzy, E], feed_dict=feed)
    # feed[z] = zy_result[:,:n_z]
    # D_result = s.run([Dzy], feed_dict=feed)[0]

    fig2=plt.figure(2,figsize=(figdim,figdim))
    fig2.clear()
    plt.suptitle('pressure vs position (%s)'%prefix)
    n = 0
    for i in range(len(p)):
        for j in range(len(v)):
            subplot(len(p),len(v),n+1)
            if n_y > 0:
                if n_y==1:
                    plt.plot(ds.data_domain, dec(enc(E_x[n]))-0.5, 'b')
                    plt.plot(ds.data_domain, dec(D_result[n])+0.5, 'r')
                else:
                    plt.plot(ds.data_domain, dec(enc(E_x[n]))-1.0, 'b')
                    plt.plot(ds.data_domain, dec(E_result[n])+0.0, 'g')
                    plt.plot(ds.data_domain, dec(D_result[n])+1.0, 'r')
                    # if n_z==1 and ds.name=='bowed3':
                    #     plt.text(0.004, -1.85, '$z$=%0.02f'%how_far[n],
                    #              horizontalalignment='right')
                if i==0 and n_y < 2:
                    plt.title('$z_0=%0.02f$\n$pos=%0.02f$'%(E_z[n,0],E_y[n,1]))
                elif i==0:
                    plt.title('$pos=%0.02f$'%v[j])
                if j==0 and n_y < 2:
                    if n_z>1:
                        plt.ylabel('$z_0=%0.02f$\n$pr=%0.02f$'%(E_z[n,1],E_y[n,0]))
                    else:
                        plt.ylabel('$pr=%0.02f$'%(E_y[n,0]))
                elif j==0:
                    plt.ylabel('$pr=%0.02f$'%p[i])
            else:
                plt.plot(ds.data_domain, dec(D_result[n]), 'k')
                if i==0: plt.title('$z_0=%0.02f$'%(E_z[n,0]))
                if j==0: plt.ylabel('$z_1=%0.02f$'%(E_z[n,1]))
            plt.xticks([])
            plt.yticks([])
            plt.ylim(-2, 2)
            plt.xlim(np.min(ds.data_domain),
                     np.max(ds.data_domain))
            n += 1
    fig2.savefig('results/'+prefix+'-varypv.pdf')
    fig2.savefig('results/'+prefix+'-varypv.png')

if False:
    fig3=plt.figure(3,figsize=(figdim,figdim))
    fig3.clear()
    #plt.suptitle('Spectra of pressure vs position')
    n = 0
    w = gaussian(np.shape(E_x[0])[0], 30)
    def plotfft(f,x,o,c):
        f = np.fft.rfftfreq(x.shape[0]*3//2, 1.0 / ds.sample_rate)
        plt.semilogx(f, (np.log(np.abs(
            np.fft.rfft(np.hstack([x*w,np.zeros(len(x)//2)])
                        + np.hstack([np.zeros(len(x)//2),x*w]))))+o), c)
        plt.xlim(0, f[-1])
    for i in range(len(p)):
        for j in range(len(v)):
            if not (i%2==1 and j%2==1):
                continue
            subplot(2,2,n+1)
            plotfft(f, dec(enc(E_x))[n], 0, 'b')
            plotfft(f, dec(E_result)[n], 4, 'g')
            plotfft(f, dec(D_result)[n], 8, 'r')
            if (i==1):
                plt.title('$pos=%0.02f$'%v[j])
            if (j==1):
                plt.ylabel('$pr=%0.02f$'%p[i])
            #plt.xticks(f[[0,-1]], f[[0,-1]])
            plt.yticks([])
            plt.ylim(-10, 15)
            n += 1
    fig3.savefig('results/'+prefix+'-specpv.pdf')
    fig3.savefig('results/'+prefix+'-specpv.png')

    # A matrix of position change and first z value
    if n_z > 0 and n_y > 0:
        r = ds.param_domain[0][1] - ds.param_domain[0][0]
        p = np.linspace(ds.param_domain[0][0]+r*0.1,
                        ds.param_domain[0][1]-r*0.1, 5)
        E_z = np.hstack([np.linspace(-0.8, 0.8, 5)]*5*n_z).reshape(-1,n_z)
        E_y = np.array([(x,32) for x in p for z in E_z[:5,0]])
        E_x, _ = ds.sample_params(E_y)
        feed = { z: E_z, x: ds.encode_data(E_x),
                 y: ds.encode_params(E_y)[:,:n_y] }
        randrot_feed(feed)
        E_result, D_result, zy_result = s.run([Dey, Dzy, E], feed_dict=feed)

        fig4=plt.figure(4,figsize=(figdim,figdim))
        fig4.clear()
        plt.suptitle('Pressure vs z (%s)'%prefix)
        n = 0
        for i in range(len(p)):
            for j in range(len(p)):
                subplot(len(p),len(p),n+1)
                plt.plot(ds.data_domain, dec(enc(E_x[n]))-0.5, 'b')
                plt.plot(ds.data_domain, dec(D_result[n])+0.5, 'r')
                if (i==0):
                    plt.title('$z_0=%0.02f$'%E_z[n,0])
                if (j==0):
                    plt.ylabel('$pr=%0.02f$'%E_y[n,0])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(-2, 2)
                plt.xlim(np.min(ds.data_domain),
                         np.max(ds.data_domain))
                n += 1
        fig4.savefig('results/'+prefix+'-varyvz.pdf')
        fig4.savefig('results/'+prefix+'-varyvz.png')

    # Analyse parameter distributions
if True:
    def regress(x,y,titl=None,hist2d=True):
        if hist2d:
            plt.hist2d(x, y, bins=40, cmap=plt.cm.magma_r)
        else:
            plt.scatter(x, y, marker='.', color='k', alpha=0.3)
        # plt.xlim(0,128)
        # plt.ylim(0,128)
        yl = plt.ylim()
        xl = plt.xlim()
        xp = xl[0] + (xl[1] - xl[0])*0.05
        yp = yl[1] - (yl[1] - yl[0])*0.02
        plt.yticks([plt.yticks()[0][0],0,plt.yticks()[0][-1]])
        plt.xticks([plt.xticks()[0][0],0,plt.xticks()[0][-1]])
        _, _, r, p, er = linregress(x,y)
        plt.text(xp, yp, '$r^2=%0.02f$\n($p %s 0.05$)'%(r**2, '><'[p<0.05]),
                 horizontalalignment='left', verticalalignment='top')
        if titl: plt.title(titl)
    def hist(x, titl):
        plt.hist(x, facecolor='w', hatch='///', edgecolor='k', linewidth=1, bins=10)
        plt.plot([-1,-1],plt.ylim(),'--k')
        plt.plot([1,1],plt.ylim(),'--k')
        plt.yticks([])
        plt.xticks([plt.xticks()[0][0],0,plt.xticks()[0][-1]])
        if titl: plt.title(titl)
    if n_y == 0 and n_z > 2:
        fig5=plt.figure(5,figsize=(figdim,figdim/2))
        fig5.clear()
        D_x, D_y = ds.sample_random(np.max([3000,ds.data.shape[0]]))
        zy_result, = s.run([E], feed_dict=randrot_feed({x: ds.encode_data(D_x)}))
        for i in range(n_z):
            subplot(n_z//2,2,i+1)
            hist(zy_result[:,i], 'Histogram of $Ez_%d$'%i)
    elif n_y == 0:
        fig5=plt.figure(5,figsize=(figdim,figdim/2))
        fig5.clear()
        D_x, D_y = ds.sample_random(np.max([3000,ds.data.shape[0]]))
        zy_result, = s.run([E], feed_dict=randrot_feed({x: ds.encode_data(D_x)}))
        subplot(221,left=False)
        hist(zy_result[:,0], 'Histogram of $Ez_0$')
        if n_z > 1:
            subplot(222)
            hist(zy_result[:,1], 'Histogram of $Ez_1$')
            subplot(223)
            regress(zy_result[:,0], zy_result[:,1],
                    '$Ez_0$ vs. $Ez_1$')
            plt.plot([1,1],[-1,1],'k--')
            plt.plot([-1,-1],[-1,1],'k--')
            plt.plot([-1,1],[1,1],'k--')
            plt.plot([-1,1],[-1,-1],'k--')
    elif n_y == 1:
        fig5=plt.figure(5,figsize=(figdim*2,figdim/3))
        fig5.clear()
        D_x, D_y = ds.sample_random(np.min([3000,ds.data.shape[0]]))
        zy_result, = s.run([E], feed_dict=randrot_feed({x: ds.encode_data(D_x)}))
        subplot(171,left=False)
        hist(zy_result[:,0], 'Histogram of $Ez_0$')
        subplot(172,left=False)
        hist(zy_result[:,n_z], 'Histogram of $Ey_0$')
        if n_z > 0:
            subplot(173)
            i = zy_result[:,0]
            j = zy_result[:,n_z]
            j = ds.decode_params(np.vstack([j, np.zeros_like(j)]).T)[:,0]
            regress(i, j, '$Ez_0$ vs. $Ey_0$')
            subplot(174)
            regress(D_y[:,0], i, '$y_0$ vs. $Ez_0$')
            subplot(175)
            regress(D_y[:,0], j, '$y_0$ vs. $Ey_0$')
            if (D_y.shape[1] > 1):
                subplot(176)
                regress(D_y[:,1], i, '$y_1$ vs. $Ez_0$')
                subplot(177)
                regress(D_y[:,1], j, '$y_1$ vs. $Ey_0$')
    elif n_y == 2:
        fig5=plt.figure(5,figsize=(figdim,figdim*2))
        fig5.clear()
        D_x, D_y = ds.sample_random(np.min([3000,ds.data.shape[0]]))
        zy_result, = s.run([E], feed_dict=randrot_feed({x: ds.encode_data(D_x)}))
        subplot(422,left=False)
        hist(zy_result[:,n_z], 'Histogram of $Ey_0$')
        subplot(423,left=False)
        hist(zy_result[:,n_z+1], 'Histogram of $Ey_1$')
        if n_z >= 1:
            subplot(421,left=False)
            hist(zy_result[:,0], 'Histogram of $Ez_0$')
            subplot(425)
            i = zy_result[:,0]
            j = zy_result[:,n_z]
            j = ds.decode_params(np.vstack([j, np.zeros_like(j)]).T)[:,0]
            regress(i, j, '$Ez_0$ vs. $Ey_0$')
            subplot(426)
            i = zy_result[:,0]
            j = zy_result[:,n_z+1]
            j = ds.decode_params(np.vstack([np.zeros_like(j), j]).T)[:,1]
            regress(i, j, '$Ez_0$ vs. $Ey_1$')
        subplot(427)
        regress(D_y[:,0],
                ds.decode_params(zy_result[:,n_z:])[:,0], '$y_0$ vs. $Ey_0$')
        subplot(428)
        regress(D_y[:,1],
                ds.decode_params(zy_result[:,n_z:])[:,1], '$y_1$ vs. $Ey_1$')
    fig5.tight_layout()
    fig5.savefig('results/'+prefix+'-paramdist.pdf')
    fig5.savefig('results/'+prefix+'-paramdist.png')

if n_z==1 and n_y==2:
    # Overlap-add synthesis
    def oa_window(p):
        feed={z: [p[:n_z]], y: [p[n_z:n_z+n_y]]}
        return ds.decode_data(s.run([Dzy], feed_dict=feed)[0][0,:])
    def vary_params1(z0,y0,N):
        D_y = np.array([np.linspace(y0[0], y0[1], N)])
        D_z = np.array([np.linspace(z0[0], z0[1], N)])
        return np.hstack([D_z.T, ds.encode_params(D_y.T)])
    def vary_params2(z0,y0,y1,N):
        D_y = np.vstack([np.linspace(y0[0], y0[1], N),
                         np.linspace(y1[0], y1[1], N)])
        D_z = np.array([np.linspace(z0[0], z0[1], N)])
        return np.hstack([D_z.T, ds.encode_params(D_y.T)])
    def plot_vary_params(p,t):
        sound = synth.overlap_add(vary_params2(*p,N=50), oa_window)
        #plt.plot(lfilter([1],[1,-0.99],x=sound))
        plt.plot(sound, 'k')
        plt.xlim(0,len(sound))
        plt.yticks([])
        plt.xticks([])
        plt.ylabel(t)
        return sound
    fig6 = plt.figure(6, figsize=(figdim*2,figdim/3))
    fig6.clear()
    subplot(311)
    sound1=plot_vary_params([[0,0],[0,128],[32,32]], 'vary $y_0$')
    subplot(312)
    sound2=plot_vary_params([[0,0],[64,64],[0,64]], 'vary $y_1$')
    subplot(313)
    sound3=plot_vary_params([[-1,1],[64,64],[32,32]], 'vary $z_0$')
    plt.xticks([0, len(sound3)], ['0', '%0.02f s'%(len(sound3)/48000.0)])
    plt.savefig('results/'+prefix+'-sound.png')
    plt.savefig('results/'+prefix+'-sound.pdf')
    import wave, struct
    with wave.open('results/'+prefix+'.wav', 'w') as wv:
        wv.setnchannels(3)
        wv.setframerate(48000)
        wv.setsampwidth(2)
        rng = np.max([sound1.max()-sound1.min(),
                      sound2.max()-sound2.min(),
                      sound3.max()-sound3.min()])
        for s1,s2,s3 in zip(sound1,sound2,sound3):
            wv.writeframes(struct.pack('<hhh', int(s1/rng*32767),
                                       int(s2/rng*32767), int(s3/rng*32767)))

if n_y == 2:
    fig7 = plt.figure(7, figsize=(figdim,figdim/2))
    fig7.clear()
    with h5py.File('known2.h5',mode='r') as k:
        # E_x, _ = ds.sample_params([k['diffed'][0,:2]])
        # plt.plot((k['diffed'][0,2:] - ds.means) / ds.stds)
        # plt.plot(E_x[0])
        if ds.name == 'bowed3':
            E_y = k['diffed'][:,:2]
            E_x = k['diffed'][:,2:]
        elif ds.mode=='normalized':
            E_y = k['diffed'][:,:2]
            E_x = (k['diffed'][:,2:] - ds.means) / ds.stds
        elif ds.mode=='rotated':
            E_y = k['periods'][:,:2]
            E_x = (k['periods'][:,2:] - ds.means) / ds.stds
        if fix_phase_for_known:
            for i in range(E_x.shape[0]):
                cor = np.zeros(100)
                # ex, _ = ds.decode_data(ds.encode_data(ds.sample_params([E_y[i,:]])))[0]
                # for c in range(100):
                #     cor[c] += np.correlate(ex, rot(ds.decode_data(ds.encode_data(E_x[i,:])), c))
                ex = ds.sample_params([E_y[i,:]])[0][0]
                for c in range(100):
                    cor[c] += np.correlate(ex, rot(E_x[i,:], c))
                E_x[i,:] = rot(E_x[i,:], cor.argmax())
        zy_result = s.run([E], feed_dict=randrot_feed({x: ds.encode_data(E_x)}))[0]
        y_estimate = ds.decode_params(zy_result[:,n_z:n_z+n_y])
        print('known y_estimate rms: ',np.sqrt(np.mean((y_estimate-E_y)**2)),
              'std:', np.sqrt(np.std((y_estimate-E_y)**2)))
        with open('results/'+prefix+'-rms.txt','a') as f:
            print('known y_estimate rms: ',np.sqrt(np.mean((y_estimate-E_y)**2)),
                  'std:', np.sqrt(np.std((y_estimate-E_y)**2)), file=f)
        plt.subplot(211)
        plt.plot(y_estimate[:,0], 'k', label='Estimated')
        plt.plot(E_y[:,0], 'k--', label='Truth')
        plt.ylabel('bow pressure $y_0$')
        plt.xticks([])
        plt.xlim(0, 48000*7/200)
        plt.ylim(ds.param_domain[0][0], ds.param_domain[0][1])
        plt.subplot(212)
        plt.plot(y_estimate[:,1], 'k', label='Estimated')
        plt.plot(E_y[:,1], 'k--', label='Truth')
        plt.ylabel('bow position $y_1$')
        plt.xticks([0,48000*7/200],[0,7])
        plt.xlim(0, 48000*7/200)
        plt.ylim(ds.param_domain[1][0], ds.param_domain[1][1])
        plt.xlabel('Time (s)')
        plt.legend(loc=3)
    fig7.tight_layout()
    fig7.savefig('results/'+prefix+'-known.pdf')
    fig7.savefig('results/'+prefix+'-known.png')

# def test_known(i=100):
#     def rot(x):
#         r = 30
#         return np.hstack([x[:,r:], x[:,:r]])
#     p = E_y[i]
#     d = rot(E_x[i:i+1])
#     cmp_X = ds.sample_params([p])
#     plt.figure(8).clear()
#     plt.plot(ds.encode_data(d).T)
#     plt.plot(ds.encode_data(cmp_X).T)
#     print('params:',p)
#     print('estimated from known:',
#           ds.decode_params(s.run([E], feed_dict={x: ds.encode_data(d)})[0]))
#     print('estimated from closest in ds:',
#           ds.decode_params(s.run([E], feed_dict={x: ds.encode_data(cmp_X)})[0]))
#     print('diff = ', np.sqrt(np.mean((cmp_X - d)**2)))
# test_known(1200)


if ds.name=='vowels':
    enc = lambda x: ds.encode_data(x)
    if ds.mode=='diffed':
        dec = lambda x: lfilter([1],[1,-0.9],x=ds.decode_data(x))
    else:
        dec = lambda x: ds.decode_data(x)

    # A matrix of position change and first z value
    if n_z > 0:
        v = np.linspace(ds.param_domain[0][0],
                        ds.param_domain[0][1], 5)
        E_z = np.zeros((5*len(v),n_z))
        E_z[:,0] = np.hstack([np.linspace(-1, 1, 7)[1:-1]]*5)
        E_y = np.array([(x,(z+1)/2*4.0) for x in v for z in E_z[:5,0]])
        E_x, _ = ds.sample_params(E_y[:,n_z:])
        feed = { z: E_z, x: ds.encode_data(E_x),
                 y: ds.encode_params(E_y)[:,:n_y] }
        randrot_feed(feed)
        E_result, D_result, zy_result = s.run([Dey, Dzy, E], feed_dict=feed)

        fig2=plt.figure(2,figsize=(figdim,figdim))
        fig2.clear()
        plt.suptitle('Vowel vs z (%s)'%prefix)
        n = 0
        for i in range(len(v)):
            for j in range(len(v)):
                subplot(len(v),len(v),n+1)
                plt.plot(ds.data_domain, dec(enc(E_x[n]))-0.5, 'b')
                plt.plot(ds.data_domain, dec(D_result[n])+0.5, 'r')
                # plt.plot(ds.data_domain, enc(E_x[n])/2-0.5*0, 'b')
                # plt.plot(ds.data_domain, D_result[n]/2+0.5*0, 'r')
                if (i==0):
                    plt.title('$z_0=%0.02f$'%E_z[j,0])
                if (j==0):
                    plt.ylabel('$v=%0.02f$'%v[i])
                plt.xticks([])
                plt.yticks([])
                plt.ylim(-2, 2)
                plt.xlim(np.min(ds.data_domain),
                         np.max(ds.data_domain))
                n += 1
        fig2.savefig('results/'+prefix+'-varyvz.pdf')
        fig2.savefig('results/'+prefix+'-varyvz.png')
    else: # n_z==0
        v = np.linspace(ds.param_domain[0][0],
                        ds.param_domain[0][1], 5)
        E_y = np.array([v]).T
        E_x, _ = ds.sample_params(E_y)
        feed = { z: [[]]*len(v), x: ds.encode_data(E_x),
                 y: ds.encode_params(E_y)[:,:n_y] }
        randrot_feed(feed)
        E_result, D_result, zy_result = s.run([Dey, Dzy, E], feed_dict=feed)

        fig2=plt.figure(2,figsize=(figdim,figdim/4))
        fig2.clear()
        plt.suptitle('Vowel vs z (%s)'%prefix)
        n = 0
        for i in range(len(v)):
            subplot(1,len(v),n+1)
            plt.plot(ds.data_domain, dec(enc(E_x[n])), 'b')
            plt.plot(ds.data_domain, dec(D_result[n]), 'r')
            # plt.plot(ds.data_domain, enc(E_x[n]), 'b')
            # plt.plot(ds.data_domain, D_result[n], 'r')
            if (j==0):
                plt.ylabel('$v=%0.02f$'%v[i])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(-2, 2)
            plt.xlim(np.min(ds.data_domain),
                     np.max(ds.data_domain))
            n += 1
        fig2.savefig('results/'+prefix+'-varyv.pdf')
        fig2.savefig('results/'+prefix+'-varyv.png')

if False:
    # Overlap-add synthesis
    def oa_window(p):
        feed={z: [p[:n_z]], y: [p[n_z:n_z+n_y]]}
        randrot_feed(feed)
        return ds.decode_data(s.run([Dzy], feed_dict=feed)[0][0,:])
    def vary_params1(z0,y0):
        D_y = np.array([np.linspace(y0[0], y0[1], 500)])
        D_z = np.array([np.linspace(z0[0], z0[1], 500)])
        return np.hstack([D_z.T, ds.encode_params(D_y.T)])
    def vary_params2(z0,y0,y1):
        D_y = np.vstack([np.linspace(y0[0], y0[1], 500),
                         np.linspace(y1[0], y1[1], 500)])
        D_z = np.array([np.linspace(z0[0], z0[1], 500)])
        return np.hstack([D_z.T, ds.encode_params(D_y.T)])
    def plot_vary_params(p,t):
        sound = synth.overlap_add(vary_params1(*p), oa_window)
        plt.plot(lfilter([1],[1,-0.99],x=sound))
        plt.xlim(0,len(sound))
        plt.yticks([])
        plt.xticks([])
        plt.title(t)
        return sound
    from importlib import reload; reload(synth)
    fig6 = plt.figure(6,figsize=(figdim,figdim/4))
    subplot(411)
    sound1=plot_vary_params([[0,0],[0,4]], 'vary $y_0$')
    subplot(412)
    sound2=plot_vary_params([[-1,1],[0,0]], 'vary $z_0$')
    subplot(413)
    sound3=plot_vary_params([[-1,1],[2,2]], 'vary $z_0$')
    subplot(414)
    sound4=plot_vary_params([[-1,1],[4,4]], 'vary $z_0$')
    plt.xticks([0, len(sound1)], [0, len(sound1)/48000.0])
    import wave, struct
    with wave.open(prefix+'.wav', 'w') as wv:
        wv.setnchannels(4)
        wv.setframerate(48000)
        wv.setsampwidth(2)
        rng = np.max([sound1.max()-sound1.min(),
                      sound2.max()-sound2.min(),
                      sound3.max()-sound3.min(),
                      sound4.max()-sound4.min()])
        for s1,s2,s3,s4 in zip(sound1,sound2,sound3,sound4):
            wv.writeframes(struct.pack('<hhhh', int(s1/rng*32767),
                                       int(s2/rng*32767),
                                       int(s3/rng*32767),
                                       int(s4/rng*32767)))

if show_plot:
    plt.show()

import gc; gc.collect()

#%% user set variables
# debug/visual stuff
show_plots = True
print_time = False # set False if running in notebook bc it already does this

# simulation setup stuff
adaptive = False # if True, use adaptive LIF neurons, else standard LIF neurons
activity_threshold = 0.02 # threshold for average pixel activity to be considered active
time_per_trial = 300 # time duration in steps of one full sequence
time_step = 1.0 # ms, temporal resolution of the simulation
learning_window = 10 # time steps of window with non-zero learning signals

# possible hyperparameters
delay = 1 # connection delay between input and recurrent neurons, min 1
tau_m = 100 # ms, time constant (leakage via e^-t/tau_m)
n_rec = 150  # number of recurrent neurons

#%% imports

import os
import time
import gc

import matplotlib.pyplot as plt
import cnsplots as cns
import tonic
import tonic.transforms as transforms

import nest
import numpy as np
import requests
from cycler import cycler
from IPython.display import Image

#%% NMNIST dataset import

train_dataset = tonic.datasets.NMNIST(save_to="./", train=True)
test_dataset = tonic.datasets.NMNIST(save_to="./", train=False) # <- Don't touch this until the final test! validate on a subset of training data as below

train_idxs = np.arange(len(train_dataset))
np.random.shuffle(train_idxs) # randomise data order

val_split = 0.1 # proportion of training data to use for validation, is not trained on
val_split_idx = int((1-val_split) * len(train_dataset))

train_idx = train_idxs[:val_split_idx]
val_idx = train_idxs[val_split_idx:]

#%% Data formatting 

# first reduce dimensions by culling inactive pixels
# this compresses the time dimension of each trial into a single frame
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)

data, labels = train_dataset[0] # initialise shape
average = frame_transform(data).astype(np.int32)

try:
     active_pixels = np.loadtxt("pixel_activations.csv", skiprows=1, delimiter=",",dtype=np.int4)
except:
    for i in range(len(train_idx)-1):
        data, labels = train_dataset[i+1]
        frame = frame_transform(data).astype(np.int32) #int16 will overflow
        average = np.add(average, frame)

    average = average[0]
    average = np.add(average[0], average[1])
    average = np.divide(average, np.max(average))

    active_pixels = np.argwhere(average > activity_threshold) # get indices of active pixels, i think doing it this way is faster and more natural than a binary mask considering the event based nature

    np.savetxt("pixel_activations.csv", X=active_pixels, header="x, y", delimiter=",",fmt='%i') # save this for later

    

if show_plots:
    fig, ax = plt.subplots()
    image = ax.imshow(average,cmap='gray')
    fig.colorbar(image)
    plt.show()

#%% Define NEST simulation setup

steps = {
    "time_per_trial": time_per_trial,  # time steps of one full sequence
    "learning_window": learning_window,  # time steps of window with non-zero learning signals
    "offset_gen": 1,  # offset since generator signals start from time step 1
    "delay_in_rec": delay,  # connection delay between input and recurrent neurons
    "extension_sim": 1,  # extra time step to close right-open simulation time interval in Simulate()
    "final_update": 3,  # extra time steps to update all synapses at the end of task
    "total_offset": 1+delay  # time steps of total offset
}

duration = {"step": time_step}  # ms, temporal resolution of the simulation

duration.update({key: value * duration["step"] for key, value in steps.items()})  # converts from time steps to ms

params_setup = {
    "print_time": print_time,  # if True, print time progress bar during simulation, set False if run as code cell
    "resolution": duration["step"],
    "total_num_virtual_procs": 4,  # number of virtual processes, set in case of distributed computing
}

nest.ResetKernel()
nest.set(**params_setup)
nest.verbosity = nest.VerbosityLevel.FATAL

#%% Define NEST neurons

active_pixels = 

n_in = active_pixels.size # number of input neurons
n_out = labels.size  # number of readout neurons, same as number of classes in dataset

if adaptive:
    model_nrn_rec = "eprop_iaf_adapt" # Adaptive LIF neuron
else:
    model_nrn_rec = "eprop_iaf" # Standard LIF neuron


params_nrn_out = {
    "C_m": 1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
    "E_L": 0.0,  # mV, resting membrane potential
    "eprop_isi_trace_cutoff": 100,  # cutoff of integration of eprop trace between spikes
    "I_e": 0.0,  # pA, external current input
    "tau_m": tau_m,  # ms, membrane time constant (leakage via e^-t/tau_m)
    "V_m": 0.0,  # mV, initial value of the membrane voltage
}

params_nrn_rec = {
    "C_m": 1.0,
    "c_reg": 2.0 / duration["sequence"],  # coefficient of firing rate regularization
    "E_L": 0.0,
    "eprop_isi_trace_cutoff": 100,
    "f_target": 10.0,  # spikes/s, target firing rate for firing rate regularization
    "flush_event_send_interval": duration[
        "sequence"
    ],  # ms, inactivity period before flushing outgoing synapses to free memory
    "I_e": 0.0,
    "kappa": 0.99,  # low-pass filter of the eligibility trace
    "kappa_reg": 0.99,  # low-pass filter of the firing rate for regularization
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": 1.0 / 1.7,  # width scaling of the pseudo-derivative
    "t_ref": 0.0,  # ms, duration of refractory period
    "tau_m": 30.0,
    "V_m": 0.0,
    "V_th": 0.6,  # mV, spike threshold membrane voltage
}

scale_factor = 1.0 - params_nrn_rec["kappa"]  # factor for rescaling due to removal of irregular spike arrival
params_nrn_rec["c_reg"] /= scale_factor**2

if model_nrn_rec == "eprop_iaf_adapt":
    params_nrn_rec["adapt_beta"] = 0.0  # adaptation scaling

if model_nrn_rec in ["eprop_iaf_psc_delta", "eprop_iaf_psc_delta_adapt"]:
    params_nrn_rec["V_reset"] = -0.5  # mV, reset membrane voltage
    params_nrn_rec["c_reg"] = 2.0 / duration["sequence"] / scale_factor**2
    params_nrn_rec["V_th"] = 0.5

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_in = nest.Create("parrot_neuron", n_in)

nrns_rec = nest.Create(model_nrn_rec, n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)
gen_learning_window = nest.Create("step_rate_generator")

# %%

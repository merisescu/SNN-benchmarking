#%% user set variables
# debug/visual stuff
show_plots = True
print_time = False # set False if running in notebook bc it already does this

# simulation setup stuff
activity_threshold = 0.02 # threshold for average pixel activity to be considered active
# if changing activity threshold delete csv file to reinitialise
time_per_trial = 300 # time duration in steps of one full sequence
time_step = 1.0 # ms, temporal resolution of the simulation
learning_window = 10 # time steps of window with non-zero learning signals
val_split = 0.1 # proportion of training data to use for validation, is not trained on
n_record = 1  # number of neurons to record dynamic variables from - this script requires n_record >= 1
n_record_w = 5  # number of senders and targets to record weights from - this script requires n_record_w >=1

n_iter_train = 20 # number of training iterations, 200 for convergence
n_iter_test = 2  # number of iterations for final test
do_early_stopping = False  # if True, stop training as soon as stop criterion fulfilled
n_iter_validate_every = 10  # number of training iterations before validation
n_iter_early_stop = 8  # number of iterations to average over to evaluate early stopping condition
stop_crit = 0.07  # error value corresponding to stop criterion for early stopping
batch_size = 128 # number of items to train on per training/validation iteration

# possible hyperparameters
adaptive = False # if True, use adaptive LIF neurons, else standard LIF neurons
delay = 1 # connection delay between input and recurrent neurons, min 1
tau_m = 100 # ms, leakage time constant (e^-t/tau_m)
V_m = 0. # mV, resting neuron (membrane) potential, also used as initial value
V_th = 0.6 # mV, threshold voltage
f_target = 10. # spikes/s, target firing rate for firing rate regularization
n_rec = 150  # number of recurrent neurons
surrogate_gradients = ["piecewise_linear",] # TODO: add the rest
# height and width scaling of pseudoderivative function
grad_height = 0.5
grad_width = 1.0 / 1.7
t_ref = 0. # ms, refractory period
beta = 1 # ALIF coefficient for scaling threshold adaptation
tau_a = 200 # ms, threshold adaptation time constant (e^-t/tau_a)

sparsity_level_in_rec = 0.75  # sparsity of synapses between input and recurrent neurons
sparsity_level_rec_rec = 0.99 # sparsity of recurrent synapses
sparsity_level_rec_out = 0.0 # sparsity of recurrent to output neurons (fully connected by default)

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
#from IPython.display import Image

#%% NMNIST dataset import

train_dataset = tonic.datasets.NMNIST(save_to="./", train=True)
test_dataset = tonic.datasets.NMNIST(save_to="./", train=False) # <- Don't touch this until the final test! validate on a subset of training data as below
num_classes = 10 # depends on dataset

test_idx = np.arange(len(test_dataset))

train_idxs = np.arange(len(train_dataset))
np.random.shuffle(train_idxs) # randomise data order

val_split_idx = int((1-val_split) * len(train_dataset))

train_idx = train_idxs[:val_split_idx]
val_idx = train_idxs[val_split_idx:]

#%% Data formatting 

# first reduce dimensions by culling inactive pixels
# this compresses the time dimension of each trial into a single frame
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=1)

data, labels = train_dataset[0] # initialise shape
average = frame_transform(data).astype(np.float32)

#%%
# TODO: change activity code to detect active pixels per polarity not combined
# load active pixels if already calculated saves time
try:
    active_pixels = np.loadtxt("pixel_activations.csv", skiprows=1, delimiter=",",dtype=[('p',np.int8),('x',np.int8),('y',np.int8)])
except:
    # collect pixel activations per trial and average
    for i in range(len(train_idx)-1):
        data, labels = train_dataset[i+1]
        frame = frame_transform(data).astype(np.float32)
        average = np.add(average, frame)

    average = average[0]
    average[0] = np.divide(average[0], np.max(average[0]))
    average[1] = np.divide(average[1], np.max(average[1]))
    
    if show_plots:
        fig, axs = plt.subplots(1,2)
        image_on = axs[0].imshow(average[1],cmap='gray')
        image_off = axs[1].imshow(average[0],cmap='gray')
        fig.colorbar(image_on)
        plt.savefig("figures/activation.eps")

    active_pixels = np.argwhere(average > activity_threshold) # get indices of active pixels, i think doing it this way is faster and more natural than a binary mask considering the event based nature

    np.savetxt("pixel_activations.csv", X=active_pixels, header="p, x, y", delimiter=",",fmt='%i')

active_pixels = active_pixels[['x','y','p']]

t_max = 336040 # TODO: remove hardcoding

print('Out of 2312 pixels: ', active_pixels.size, "are active")
print(len(active_pixels))
# test data formatting

item = 0
data = train_dataset[item][0]
#print(data[['x','y']])
#print(data.dtype)
#print(active_pixels[0].dtype)
images_group = [[],[],[]]
t_max = 336040
n_in = active_pixels.size # number of input neurons (*2 because 2 polarities)
for px in range(n_in):
    mask = np.isin(data[['x','y','p']],active_pixels[px])
    relative_times = np.around(data[mask]["t"] * time_per_trial / t_max)
    images_group[item].extend([relative_times])
    #if len(relative_times) >= 1:
    #    print(relative_times)
print(images_group[0])

def get_params_task_input_output(n_iter_interval, input_group, target_group):

    spike_times = [[] for _ in range(n_in)]

    iteration_offset = n_iter_interval * batch_size * duration["time_per_trial"]

    params_gen_rate_target = [
        {
            "amplitude_times": np.arange(0.0, batch_size * duration["time_per_trial"], duration["time_per_trial"])
            + iteration_offset
            + duration["total_offset"],
            "amplitude_values": np.zeros(batch_size),
        }
        for _ in range(n_out)
    ]

    for group_element in range(batch_size):
        params_gen_rate_target[target_group[group_element]]["amplitude_values"][group_element] = 1.0
        for n, relative_times in enumerate(input_group[group_element]):
            if len(relative_times) > 0:
                #print(n, relative_times)
                relative_times = np.array(relative_times)
                spike_times[n].extend(
                    iteration_offset + group_element * duration["time_per_trial"] + relative_times + duration["offset_gen"]
                )
                # n is the pixel channel i think (flattened to 1d)
                # relative spike times for each item is converted to absolute time in simulation
                # iteration_offset - absolute amount of time since simulation start i.e. number of simulation trials (training, validation etc.) * size of each trial (group size * duration of each trial), then add group_element*duration['sequence'] to add how far along we are in this trial, then the offset_gen for the initial timer zeroing, ALL added to the relative time

    params_gen_learning_window = {
        "amplitude_times": np.hstack(
            [
                np.array([0.0, duration["time_per_trial"] - duration["learning_window"]])
                + iteration_offset
                + group_element * duration["time_per_trial"]
                + duration["total_offset"]
                for group_element in range(batch_size)
            ]
        ),
        "amplitude_values": np.tile([0.0, 1.0], batch_size),
    }

    params_gen_spk_in = [{"spike_times": spk_times} for spk_times in spike_times]

    return params_gen_spk_in, params_gen_rate_target, params_gen_learning_window

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

n_in = active_pixels.size # number of input neurons (*2 because 2 polarities)
n_out = num_classes  # number of readout neurons, same as number of classes in dataset

if adaptive:
    model_nrn_rec = "eprop_iaf_adapt" # Adaptive LIF neuron
else:
    model_nrn_rec = "eprop_iaf" # Standard LIF neuron


params_nrn_out = {
    "C_m": 1.0,  # (not used) pF, membrane capacitance - takes effect only if neurons get current input
    "E_L": V_m,  # mV, resting membrane potential
    "eprop_isi_trace_cutoff": 100,  # cutoff of integration of eprop trace between spikes
    "I_e": 0.0,  # (not used) pA, external current input 
    "tau_m": tau_m,  # ms, membrane time constant (leakage via e^-t/tau_m)
    "V_m": V_m,  # mV, initial value of the membrane voltage
}

params_nrn_rec = {
    "C_m": 1.0,  # (not used)
    "c_reg": 2.0 / duration["time_per_trial"],  # coefficient of firing rate regularization
    "E_L": V_m,
    "eprop_isi_trace_cutoff": 100,
    "f_target": f_target,  # spikes/s, target firing rate for firing rate regularization
    "flush_event_send_interval": duration[
        "time_per_trial"
    ],  # ms, inactivity period before flushing outgoing synapses to free memory
    "I_e": 0.0, # (not used)
    "kappa": 0.99,  # low-pass filter of the eligibility trace
    "kappa_reg": 0.99,  # low-pass filter of the firing rate for regularization
    "surrogate_gradient_function": surrogate_gradients[0],  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": grad_height,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": grad_width,  # width scaling of the pseudo-derivative
    "t_ref": 0.0,  # ms, duration of refractory period
    "tau_m": tau_m,
    "V_m": V_m,
    "V_th": V_th,  # mV, spike threshold membrane voltage
}

scale_factor = 1.0 - params_nrn_rec["kappa"]  # factor for rescaling due to removal of irregular spike arrival
params_nrn_rec["c_reg"] /= scale_factor**2

if model_nrn_rec == "eprop_iaf_adapt":
    params_nrn_rec["adapt_beta"] = beta  # adaptation scaling
    params_nrn_rec["adapt_tau"] = tau_a # ms, adaptation time constant

if model_nrn_rec in ["eprop_iaf_psc_delta", "eprop_iaf_psc_delta_adapt"]:
    params_nrn_rec["V_reset"] = -0.5  # mV, reset membrane voltage
    params_nrn_rec["c_reg"] = 2.0 / duration["time_per_trial"] / scale_factor**2
    params_nrn_rec["V_th"] = 0.5

# Intermediate parrot neurons required between input spike generators and recurrent neurons,
# since devices cannot establish plastic synapses for technical reasons

gen_spk_in = nest.Create("spike_generator", n_in)
nrns_in = nest.Create("parrot_neuron", n_in)

nrns_rec = nest.Create(model_nrn_rec, n_rec, params_nrn_rec)
nrns_out = nest.Create("eprop_readout", n_out, params_nrn_out)
gen_rate_target = nest.Create("step_rate_generator", n_out)
gen_learning_window = nest.Create("step_rate_generator")

#%% Define recorders

if n_record == 0 or n_record_w == 0:
    raise ValueError("n_record and n_record_w >= 1 required")

params_mm_rec = {
    "interval": duration["step"],  # interval between two recorded time points
    "record_from": ["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
    "start": duration["offset_gen"] + duration["delay_in_rec"],  # start time of recording
    "label": "multimeter_rec",
}

params_mm_out = {
    "interval": duration["step"],
    "record_from": ["V_m", "readout_signal", "target_signal", "error_signal"],
    "start": duration["total_offset"],
    "label": "multimeter_out",
}

params_wr = {
    "senders": nrns_in[:n_record_w] + nrns_rec[:n_record_w],  # limit senders to subsample weights to record
    "targets": nrns_rec[:n_record_w] + nrns_out,  # limit targets to subsample weights to record from
    "start": duration["total_offset"],
    "label": "weight_recorder",
}

params_sr_in = {
    "start": duration["offset_gen"],
    "label": "spike_recorder_in",
}

params_sr_rec = {
    "start": duration["offset_gen"],
    "label": "spike_recorder_rec",
}

mm_rec = nest.Create("multimeter", params_mm_rec)
mm_out = nest.Create("multimeter", params_mm_out)
sr_in = nest.Create("spike_recorder", params_sr_in)
sr_rec = nest.Create("spike_recorder", params_sr_rec)
wr = nest.Create("weight_recorder", params_wr)

nrns_rec_record = nrns_rec[:n_record]

#%% Define synapses and connections w recorders

params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False} # autapses: recurrent connections to self neuron
params_conn_one_to_one = {"rule": "one_to_one"}


def calculate_glorot_dist(fan_in, fan_out):
    glorot_scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    glorot_limit = np.sqrt(3.0 * glorot_scale)
    glorot_distribution = np.random.uniform(low=-glorot_limit, high=glorot_limit, size=(fan_in, fan_out))
    return glorot_distribution


def create_mask(weights, sparsity_level):
    return np.random.choice([0, 1], weights.shape, p=[sparsity_level, 1 - sparsity_level])


def get_weight_recorder_senders_targets(weights, sender_pop, target_pop):
    target_idc, sender_idc = np.where(weights)
    senders = sender_pop[np.unique(sender_idc[:n_record_w])]
    targets = target_pop[np.unique(target_idc[:n_record_w])]
    return senders, targets


weights_in_rec = np.array(np.random.randn(n_in, n_rec).T / np.sqrt(n_in))
weights_rec_rec = np.array(np.random.randn(n_rec, n_rec).T / np.sqrt(n_rec))
np.fill_diagonal(weights_rec_rec, 0.0)  # since no autapses set corresponding weights to zero
weights_rec_out = np.array(calculate_glorot_dist(n_rec, n_out).T) * scale_factor
weights_out_rec = np.array(np.random.randn(n_rec, n_out)) / scale_factor

weights_in_rec *= create_mask(weights_in_rec, sparsity_level_in_rec)
weights_rec_rec *= create_mask(weights_rec_rec, sparsity_level_rec_rec)
weights_rec_out *= create_mask(weights_rec_out, sparsity_level_rec_out)

senders_in_rec, targets_in_rec = get_weight_recorder_senders_targets(weights_in_rec, nrns_in, nrns_rec)
senders_rec_rec, targets_rec_rec = get_weight_recorder_senders_targets(weights_rec_rec, nrns_rec, nrns_rec)
senders_rec_out, targets_rec_out = get_weight_recorder_senders_targets(weights_rec_out, nrns_rec, nrns_out)

wr.set(
    senders=np.unique(np.concatenate([senders_in_rec, senders_rec_rec, senders_rec_out])),
    targets=np.unique(np.concatenate([targets_in_rec, targets_rec_rec, targets_rec_out])),
)

params_common_syn_eprop = {
    "optimizer": {
        "type": "gradient_descent",  # algorithm to optimize the weights TODO: give options
        "batch_size": 1, # TODO: implement batch here and not in series
        "optimize_each_step": False,  # call optimizer every time step (True) or once per spike (False); both
        # yield same results for gradient descent, False offers speed-up
        "Wmin": -100.0,  # pA, minimal limit of the synaptic weights
        "Wmax": 100.0,  # pA, maximal limit of the synaptic weights
    },
    "weight_recorder": wr,
}

eta_test = 0.0  # learning rate for test phase
eta_train = 5e-3 * scale_factor**2  # learning rate for training phase

params_syn_base = {
    "synapse_model": "eprop_synapse",
    "delay": duration["step"],  # ms, dendritic delay
}

params_syn_in = params_syn_base.copy()
params_syn_rec = params_syn_base.copy()
params_syn_out = params_syn_base.copy()

params_syn_feedback = {
    "synapse_model": "eprop_learning_signal_connection",
    "delay": duration["step"],
    "weight": weights_out_rec,
}

params_syn_learning_window = {
    "synapse_model": "rate_connection_delayed",
    "delay": duration["step"],
    "receptor_type": 1,  # receptor type over which readout neuron receives learning window signal
}

params_syn_rate_target = {
    "synapse_model": "rate_connection_delayed",
    "delay": duration["step"],
    "receptor_type": 2,  # receptor type over which readout neuron receives target signal
}

params_syn_static = {
    "synapse_model": "static_synapse",
    "delay": duration["step"],
}

nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

nest.Connect(gen_spk_in, nrns_in, params_conn_one_to_one, params_syn_static)  # connection 1


def sparsely_connect(weights, params_syn, nrns_pre, nrns_post):
    targets, sources = np.where(weights)
    params_syn["weight"] = weights[targets, sources].flatten()
    params_syn["delay"] = [params_syn["delay"] for _ in params_syn["weight"]]
    nrns_pre_arr = np.array(nrns_pre.tolist())
    nrns_post_arr = np.array(nrns_post.tolist())
    nest.Connect(nrns_pre_arr[sources], nrns_post_arr[targets], params_conn_one_to_one, params_syn)


sparsely_connect(weights_in_rec, params_syn_in, nrns_in, nrns_rec)  # connection 2
sparsely_connect(weights_rec_rec, params_syn_rec, nrns_rec, nrns_rec)  # connection 3
sparsely_connect(weights_rec_out, params_syn_out, nrns_rec, nrns_out)  # connection 4

nest.Connect(nrns_out, nrns_rec, params_conn_all_to_all, params_syn_feedback)  # connection 5
nest.Connect(gen_rate_target, nrns_out, params_conn_one_to_one, params_syn_rate_target)  # connection 6
nest.Connect(gen_learning_window, nrns_out, params_conn_all_to_all, params_syn_learning_window)  # connection 7

nest.Connect(nrns_in, sr_in, params_conn_all_to_all, params_syn_static)
nest.Connect(nrns_rec, sr_rec, params_conn_all_to_all, params_syn_static)

nest.Connect(mm_rec, nrns_rec_record, params_conn_all_to_all, params_syn_static)
nest.Connect(mm_out, nrns_out, params_conn_all_to_all, params_syn_static)

# Send a final update spike to force update to all neurons

gen_spk_final_update = nest.Create("spike_generator", 1)

nest.Connect(gen_spk_final_update, nrns_in + nrns_rec, "all_to_all", {"weight": 1000.0})

#%% Store initial weights

def get_weights(pop_pre, pop_post):
    conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
    conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
    conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])

    conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
    conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
    return conns


weights_pre_train = {
    "in_rec": get_weights(nrns_in, nrns_rec),
    "rec_rec": get_weights(nrns_rec, nrns_rec),
    "rec_out": get_weights(nrns_rec, nrns_out),
}

#%% Simulate
# TODO: Finish migrating data loader to training pipeline
class TrainingPipeline:
    def __init__(self):
        self.results_dict = {
            "error": [],
            "loss": [],
            "iteration": [],
            "label": [],
        }
        self.n_iter_sim = 0
        self.phase_label_previous = ""
        self.error = 0
        self.k_iter = 0
        self.early_stop = False

        self.current_index = 0

    def get_batch(self,dataset,idx):
        end_index = self.current_index + batch_size

        selected_indices = np.take(idx, range(self.current_index, end_index), mode="wrap")

        self.current_index = (self.current_index + batch_size) % len(idx)

        # filter out inactive pixels and return as flattened array n, relative_times
        # TODO: switch to hashing pixels
        images_group = [[] for _ in selected_indices]
        #print(selected_indices)
        #print(len(images_group))
        for item in range(len(selected_indices)):
            data = dataset[selected_indices[item]][0]
            for px in range(n_in):
                mask = np.isin(data[['x','y','p']],active_pixels[px])
                relative_times = np.around(data[mask]["t"] * time_per_trial / t_max)
                images_group[item].extend([relative_times])
                #if len(relative_times) >= 1:
                #    print(relative_times)

        labels_group = [dataset[item][1] for item in selected_indices]

        return images_group, labels_group

    def evaluate(self):
        events_mm_out = mm_out.get("events")

        readout_signal = events_mm_out["readout_signal"]
        target_signal = events_mm_out["target_signal"]
        senders = events_mm_out["senders"]
        times = events_mm_out["times"]

        cond1 = times > (self.n_iter_sim - 1) * batch_size * duration["time_per_trial"] + duration["total_offset"]
        cond2 = times <= self.n_iter_sim * batch_size * duration["time_per_trial"] + duration["total_offset"]
        idc = cond1 & cond2

        readout_signal = np.array([readout_signal[idc][senders[idc] == i] for i in np.unique(senders)])
        target_signal = np.array([target_signal[idc][senders[idc] == i] for i in np.unique(senders)])

        readout_signal = readout_signal.reshape((n_out, 1, batch_size, steps["time_per_trial"]))
        target_signal = target_signal.reshape((n_out, 1, batch_size, steps["time_per_trial"]))

        readout_signal = readout_signal[:, :, :, -steps["learning_window"] :]
        target_signal = target_signal[:, :, :, -steps["learning_window"] :]

        loss = 0.5 * np.mean(np.sum((readout_signal - target_signal) ** 2, axis=3), axis=(0, 2))

        y_prediction = np.argmax(np.mean(readout_signal, axis=3), axis=0)
        y_target = np.argmax(np.mean(target_signal, axis=3), axis=0)
        accuracy = np.mean((y_target == y_prediction), axis=1)
        errors = 1.0 - accuracy

        self.results_dict["iteration"].append(self.n_iter_sim)
        self.results_dict["error"].extend(errors)
        self.results_dict["loss"].extend(loss)
        self.results_dict["label"].append(self.phase_label_previous)

        self.error = errors[0]

    def run_phase(self, phase_label, eta, images, labels):
        params_common_syn_eprop["optimizer"]["eta"] = eta
        nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

        params_gen_spk_in, params_gen_rate_target, params_gen_learning_window = get_params_task_input_output(
            self.n_iter_sim, images, labels
        )
        gen_spk_in.set(params_gen_spk_in)
        gen_rate_target.set(params_gen_rate_target)
        gen_learning_window.set(params_gen_learning_window)

        self.simulate("total_offset")
        self.simulate("extension_sim")

        if self.n_iter_sim > 0:
            self.evaluate()

        duration["sim"] = batch_size * duration["time_per_trial"] - duration["total_offset"] - duration["extension_sim"]

        self.simulate("sim")

        self.n_iter_sim += 1
        self.phase_label_previous = phase_label

    def run_training(self):
        images, labels = self.get_batch(train_dataset, train_idx)
        self.run_phase("training", eta_train, images, labels)

    def run_validation(self):
        images, labels = self.get_batch(train_dataset, val_idx)
        if do_early_stopping and self.k_iter % n_iter_validate_every == 0:
            self.run_phase("validation", eta_test, images, labels)

    def run_early_stopping(self):
        if do_early_stopping and self.k_iter % n_iter_validate_every == 0:
            if self.k_iter > 0 and self.error < stop_crit:
                errors_early_stop = []
                for _ in range(n_iter_early_stop):
                    images, labels = self.get_batch(train_dataset, val_idx)
                    self.run_phase("early-stopping", eta_test, images, labels)
                    errors_early_stop.append(self.error)

                self.early_stop = np.mean(errors_early_stop) < stop_crit

    def run_test(self):
        images, labels = self.get_batch(test_dataset, test_idx)
        for _ in range(n_iter_test):
            self.run_phase("test", eta_test, images, labels)

    def simulate(self, k):
        nest.Simulate(duration[k])

    def run(self):
        while self.k_iter < n_iter_train and not self.early_stop:
            self.run_validation()
            self.run_early_stopping()
            self.run_training()
            self.k_iter += 1

        self.run_test()

        self.simulate("total_offset")
        self.simulate("extension_sim")

        self.evaluate()

        duration["task"] = self.n_iter_sim * batch_size * duration["time_per_trial"] + duration["total_offset"]

        gen_spk_final_update.set({"spike_times": [duration["task"] + duration["extension_sim"] + 1]})

        self.simulate("final_update")

    def get_results(self):
        for k, v in self.results_dict.items():
            self.results_dict[k] = np.array(v)
        return self.results_dict


training_pipeline = TrainingPipeline()
training_pipeline.run()

results_dict = training_pipeline.get_results()
n_iter_sim = training_pipeline.n_iter_sim

#%% Record post training data

weights_post_train = {
    "in_rec": get_weights(nrns_in, nrns_rec),
    "rec_rec": get_weights(nrns_rec, nrns_rec),
    "rec_out": get_weights(nrns_rec, nrns_out),
}

events_mm_rec = mm_rec.get("events")
events_mm_out = mm_out.get("events")
events_sr_in = sr_in.get("events")
events_sr_rec = sr_rec.get("events")
events_wr = wr.get("events")

#%% Visualisation
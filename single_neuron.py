# a simple script to compare LIF and ALIF neuron traces

#%% LIF
import nest
import numpy as np
import matplotlib.pyplot as plt

steps = 800
resolution = 1 # ms
duration = steps*resolution
# neuron parameters
eprop_isi_trace_cutoff = 100 # after this many steps after a spike, stops computing eligibility trace
params_nrn_out = {
    "tau_m": 100,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "V_m": 0.,
    "E_L": 0.
}

params_nrn_rec = {
    "tau_m": 30,
    "c_reg": 2,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "E_L": 0.,
    "V_m": 0.,
    "V_th": 0.6,
    "t_ref": 0,
    "kappa": 0.99,
    "kappa_reg": 0.99,
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": 1.0 / 1.7,  # width scaling of the pseudo-derivative
    "f_target": 10, # Hz
    #"I_e":100, # pA
    #"C_m": 250, # pF
}

params_mm_rec = {
            "interval": resolution,  # interval between two recorded time points
            "record_from": ["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
            "start": 0,  # start time of recording
            "label": "multimeter_rec",
        }

params_sr_rec = {
            "start": 0,
            "label": "spike_recorder_rec",
        }

params_syn_static = {
            "synapse_model": "static_synapse"
        }
amplitude_times = [1,100,600]
amplitude_values = [0,10,0]
params_step = {
    "amplitude_times": amplitude_times,
    "amplitude_values": amplitude_values
}
times = np.arange(0, duration, resolution)
current = np.zeros(len(times))
for t, value in zip(amplitude_times, amplitude_values):
    current[times >= t] = value

scale_factor = 1.0 - params_nrn_rec["kappa"]
params_nrn_rec["c_reg"] /= duration
params_nrn_rec["c_reg"] /= scale_factor**2

nest.ResetKernel()
nest.verbosity = nest.VerbosityLevel.FATAL
params_setup = {"resolution":resolution}
nest.set(**params_setup)

input = nest.Create("step_current_generator",1,params_step)
neuron = nest.Create("eprop_iaf",1,params_nrn_rec)

mm = nest.Create("multimeter",params_mm_rec)
sr = nest.Create("spike_recorder",params_sr_rec)

nest.Connect(input,neuron,{"rule": "one_to_one"},params_syn_static)
nest.Connect(mm,neuron,{"rule": "one_to_one"},params_syn_static)
nest.Connect(neuron,sr,{"rule": "one_to_one"},params_syn_static)

nest.Simulate(duration)

events_mm = mm.get('events')
events_sr = sr.get('events')
v_m = events_mm['V_m']
time = events_mm['times']
spikes = events_sr['times']
for i in spikes:
    i = int(i)
    v_m[i-1]+=3 # arbitrary value just big enough to clearly be much larger than the subthreshold dynamics
fig, ax = plt.subplots(2,1)
ax[0].plot(time,current[:-1])
ax[0].set_title("Input step current")
ax[0].set_ylabel(r"$I_{ext}$")
ax[1].plot(time, v_m)
ax[1].set_title("Adaptive neuron voltage dynamics")
ax[1].set_ylabel(r"$V_j$")
fig.tight_layout()
#%% adaptive LIF

params_nrn_rec["adapt_beta"] = 0.1
params_nrn_rec["adapt_tau"] = 600

nest.ResetKernel()
nest.verbosity = nest.VerbosityLevel.FATAL
nest.set(**params_setup)

input = nest.Create("step_current_generator",1,params_step)
neuron = nest.Create("eprop_iaf_adapt",1,params_nrn_rec)

params_mm_rec['record_from'] = ["V_m", "surrogate_gradient", "learning_signal", "V_th_adapt"]
mm = nest.Create("multimeter",params_mm_rec)
sr = nest.Create("spike_recorder",params_sr_rec)

nest.Connect(input,neuron,{"rule": "one_to_one"},params_syn_static)
nest.Connect(mm,neuron,{"rule": "one_to_one"},params_syn_static)
nest.Connect(neuron,sr,{"rule": "one_to_one"},params_syn_static)

nest.Simulate(duration)

events_mm = mm.get('events')
events_sr = sr.get('events')
v_m = events_mm['V_m']
v_th = events_mm['V_th_adapt']
time = events_mm['times']
spikes = events_sr['times']

for i in spikes:
    i = int(i)
    v_m[i-1]+=3

v_m_adapt = v_m - (v_th-0.6)
fig, ax = plt.subplots(3,1)
ax[0].plot(time,current[:-1])
ax[0].set_title("Input step current")
ax[0].set_ylabel(r"$I_{ext}$")
ax[1].plot(time, v_m)
ax[1].set_title("Adaptive neuron voltage dynamics")
ax[1].set_ylabel(r"$V_j$")
ax[2].plot(time, v_m_adapt)
ax[2].set_title('Voltage correcting for threshold adaptation ("Effective voltage")')
ax[2].set_ylabel(r"$V_j-\beta_aa_j$")
ax[2].set_xlabel(r"time (ms)")
ax[2].axhline(0.6,linestyle='--',linewidth=0.5,color='k')
fig.tight_layout()

# %% ALIF with rate spike input

nest.ResetKernel()
nest.verbosity = nest.VerbosityLevel.FATAL
nest.set(**params_setup)

params_nrn_rec["adapt_beta"] = 0.1
params_nrn_rec["adapt_tau"] = 600

spike_times = []
spike_weights = []
in_spikes = np.zeros(799)
for time in np.arange(100,600,8):
    spike_times.append(time)
    spike_weights.append(0.3)
    in_spikes[time] = 1

params_spikes_in = {"spike_times":spike_times,"spike_weights":spike_weights}

input = nest.Create("spike_generator",1,params_spikes_in)
neuron = nest.Create("eprop_iaf_adapt",1,params_nrn_rec)

params_mm_rec['record_from'] = ["V_m", "surrogate_gradient", "learning_signal", "V_th_adapt"]
mm = nest.Create("multimeter",params_mm_rec)
sr = nest.Create("spike_recorder",params_sr_rec)

nest.Connect(input,neuron,{"rule": "one_to_one"},params_syn_static)
nest.Connect(mm,neuron,{"rule": "one_to_one"},params_syn_static)
nest.Connect(neuron,sr,{"rule": "one_to_one"},params_syn_static)

nest.Simulate(duration)


events_mm = mm.get('events')
events_sr = sr.get('events')
v_m = events_mm['V_m']
v_th = events_mm['V_th_adapt']
time = events_mm['times']
spikes = events_sr['times']

for i in spikes:
    i = int(i)
    v_m[i-1]+=3

v_m_adapt = v_m - (v_th-0.6)
fig, ax = plt.subplots(3,1)
ax[0].plot(time,in_spikes)
ax[0].set_title("Contstant rate neuron spike train")
ax[0].set_ylabel(r"$z_i$")
ax[1].plot(time, v_m)
ax[1].set_title("Adaptive neuron voltage dynamics")
ax[1].set_ylabel(r"$v_j$")
ax[2].plot(time, v_m_adapt)
ax[2].set_title('Voltage correcting for threshold adaptation ("Effective voltage")')
ax[2].set_ylabel(r"$v_j-\beta_aa_j$")
ax[2].set_xlabel(r"time (ms)")
ax[2].axhline(0.6,linestyle='--',linewidth=0.5,color='k')
fig.tight_layout()

# %% Izhikevich

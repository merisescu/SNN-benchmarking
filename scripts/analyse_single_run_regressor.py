#%% imports
import numpy as np
import scripts.api.setup as setup
#from reservoirpy.datasets import mackey_glass


#%% func definition

# Using 1D Mackey-Glass
func_max = 1.32
# with standard parameters
lyapunov_time = 197
#resolution = (lyapunov_time / 75) / 10
resolution = 1
#n_timesteps = round( 20 * lyapunov_time / resolution) # (20 * 75)
n_timesteps = 1500
# keeping for clarity of calculation
'''def objective_func(steps):
    f = mackey_glass(steps).T
    f /= func_max
    return f
print(n_timesteps)'''
def mackey_glass(
    n_timesteps = None,
    duration=20 * lyapunov_time,
    dt=resolution,
    tau=17,
    beta=0.2,
    gamma=0.1,
    n=10,
    x0=1.2,
):
    if n_timesteps:
        n_steps = n_timesteps
    else:
        n_steps = int(duration / dt)

    delay_steps = round(tau / dt)

    # History + future
    x = np.full(n_steps + delay_steps + 1, x0, dtype=np.float64)

    for t in range(delay_steps, n_steps + delay_steps):
        delayed = x[t - delay_steps]

        dx = beta * delayed / (1.0 + delayed**n) - gamma * x[t]

        x[t + 1] = x[t] + dt * dx
    return x[delay_steps+1:]/func_max

def objective_func(steps):
    return [mackey_glass(n_timesteps=steps,dt=resolution)]
#%% set parameters

# simulation parameters
steps = {
    "sequence": n_timesteps,  # time steps of one full sequence
}

# neuron parameters
eprop_isi_trace_cutoff = n_timesteps # after this many steps after a spike, stops computing eligibility trace
params_nrn_out = {
    "tau_m": 24.5,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "V_m": 0.,
    "E_L": 0.
}


'''params_nrn_rec = {
    "tau_m": 142,
    "c_reg": 3.3,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "E_L": 0.,
    "V_m": 0.,
    "V_th": 1.58,
    "t_ref": 2.55,
    "kappa": 0.92,
    "kappa_reg": 0.97,
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": 1,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": 1,  # width scaling of the pseudo-derivative

    "adapt_beta": 1.43,
    "adapt_tau": 156,

    "f_target": 1.8, # Hz
}'''

#izhikevich
'''params_nrn_rec = {
    "c_reg": 30,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "V_th": 30,
    "kappa": 0.99,
    "kappa_reg": 0.97,
    "surrogate_gradient_function": "fast_sigmoid_derivative",  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": 1,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": 0.5,  # width scaling of the pseudo-derivative

    "f_target": 1, # Hz
}'''
a = 0.1
b = 0.25
v_m = np.min(np.roots([0.04,5-b,140]))
u_m = b * v_m

params_nrn_rec = {
    "a": a,
    "b": b,
    "V_m": v_m,
    "U_m": u_m,
    "c_reg": 30,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "V_th": 30,
    "kappa": 0.97,
    "kappa_reg": 0.97,
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": 1/1.7,  # width scaling of the pseudo-derivative

    "f_target": 1, # Hz
}


#%% work


model = setup.eprop_model(mode='Regressor',num_threads=1,num_virtual_procs=4)
model.timing(1,1,steps)
model.input_layer(1,encoding='delta',channels=14)
model.recurrent_layer(175,params_nrn_rec,use_izhikevich=True,scale=22)
model.output_layer(1,params_nrn_out)

#%%
model.recorders(profiling='full',save_to_disk=False)
model.connect(sparsity=[0.8,0.8,0],eta_train=5e-3,optimizer='gradient_descent')
weights_pre_train = model.compile()
model.fit(function=objective_func,n_iter_train=200,group_size=1,n_validations=1,testing=True,prune=False) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

#%% debug

print(model.fmin, model.fmax)

#%%

results, events = model.get_results()

smape = results['error']
spikes = results['spikes']

#%% test loading info
import os
import matplotlib.pyplot as plt


#data_dir = "./data/"
#plots_dir = "./figures/"

#timestamps = os.listdir(data_dir)
#print(timestamps)
#timestamp = timestamps[0]

events_mm_rec = events['mm_rec']
events_mm_out = events['mm_out']
events_sr_in = events['sr_in']
events_sr_rec = events['sr_rec']
events_wr = events['wr']
weights_before = weights_pre_train
weights_after = {
                "in_rec": model.get_weights(model.nrns_in, model.nrns_rec),
                "rec_rec": model.get_weights(model.nrns_rec, model.nrns_rec),
                "rec_out": model.get_weights(model.nrns_rec, model.nrns_out),
            }
weights_post_train = weights_after

#%% Visualisation
from cycler import cycler
#os.makedirs(f"{plots_dir}{timestamp}", exist_ok=True)

#TODO: switch to cnsplots

colors = {
    "blue": "#2854c5ff",
    "red": "#e04b40ff",
    "purple": "#984EA3",
    "green": "#25aa2cff",
    "gold": "#f9c643ff",
    "white": "#ffffffff",
}

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.prop_cycle": cycler(color=[colors[k] for k in ["blue", "red", "green", "gold"]]),
    }
)

#%% learning over time

def averages_std_over_time(input, window_size):

    tail = int(np.floor(window_size/2))
    ma = np.zeros(len(input))
    ma[:tail+1] = input[:tail+1]
    ma[-tail-1:] = input[-tail-1:]
    ma[tail:-tail] = np.convolve(input,np.ones(window_size)/window_size,mode='valid')
    return ma

fig, axs = plt.subplots(3, 1, sharex=True)
if model.adaptive:
    fig.suptitle("Adaptive LIF model performance")
else:
    fig.suptitle("LIF model performance")

#axs[1].axhline(0.14,linestyle='--',linewidth=0.5,color='k',label=r'target')

loss_ma = averages_std_over_time(results["loss"][:-2],3)
acc_ma = averages_std_over_time(results["error"][:-2],3)
#axs[0].plot(results["iteration"][:-2], loss_ma, label='training')
#axs[1].plot(results["iteration"][:-2], acc_ma, label='training')

axs[0].plot(results["iteration"][:-1], results["loss"][:-1], label='training')
axs[1].plot(results["iteration"][:-1], results["error"][:-1], label='training')
'''
for color, label in zip(colors, set(results_dict["label"])):
    idc = results_dict["label"] == label
    #axs[0].scatter(results_dict["iteration"][idc], results_dict["loss"][idc], label=label)
    axs[1].scatter(results_dict["iteration"][idc], 1-results_dict["error"][idc], label=label)
'''

axs[0].scatter(results["iteration"][-1],np.mean(results["loss"][-1:]),label='test')
axs[1].scatter(results["iteration"][-1],np.mean(results["error"][-1:]),label='test')
axs[0].set_ylabel(r"$\mathcal{L} = \frac{1}{2} \sum_{t,k} \left( y_k^t -y_k^{*,t}\right)^2$")
axs[1].axhline(0.01,c='k',linewidth=0.5,linestyle='--')
axs[1].set_ylim([0.005,0.1])
axs[1].set_ylabel(r"one-step prediction")

'''spikes = np.zeros(len(results['spikes']))

for i in range(len(results['spikes'])-1):
    spikes[i+1] = results['spikes'][i+1]
spikes = spikes'''
axs[2].plot(results["iteration"][:-1],results["spikes"][:-1])
axs[2].scatter(results["iteration"][-1],results["spikes"][-1])
axs[2].set_ylabel("spikes/inf.")

axs[-1].set_xlabel("epoch")
axs[-1].legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
axs[-1].xaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()

#fig.savefig(f"{plots_dir}{timestamp}/learning_over_time.eps")

#%% spiking variables

chosen = model.nrns_rec_record.get("global_id")
nrns_in = model.nrns_in.get("global_id")

def plot_recordable(ax, events, recordable, ylabel, xlims, overlay=False):
    for sender in np.unique(events["senders"]):
        idc_sender = events["senders"] == sender
        idc_times = (events["times"][idc_sender] > xlims[0]) & (events["times"][idc_sender] < xlims[1])
        ax.plot(events["times"][idc_sender][idc_times], events[recordable][idc_sender][idc_times], lw=0.5)
    if not overlay:
        ax.set_ylabel(ylabel)
        margin = np.abs(np.max(events[recordable][idc_sender][idc_times]) - np.min(events[recordable][idc_sender][idc_times])) * 0.1
        ax.set_ylim(np.min(events[recordable][idc_sender][idc_times]) - margin, np.max(events[recordable][idc_sender][idc_times]) + margin)

def plot_spikes(ax, events, ylabel, xlims):
    idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
    senders_subset = events["senders"][idc_times]
    times_subset = events["times"][idc_times]
    chosen_senders = np.asarray(senders_subset)
    chosen_senders = np.argwhere(chosen_senders==chosen)
    chosen_times = np.asarray(times_subset)
    chosen_times = chosen_times[chosen_senders]
    chosen_r = np.ones(len(chosen_times))*chosen

    ax.scatter(times_subset, senders_subset, s=0.2)
    ax.scatter(chosen_times, chosen_r, s=0.2, c = 'r')
    ax.set_ylabel(ylabel)
    margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1
    ax.set_ylim(np.min(senders_subset) - margin, np.max(senders_subset) + margin)
    ax.set_yticks([])
    ax.set_yticklabels([])

def plot_spike_rate(ax, events, ylabel, xlims):
    idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
    #senders_subset = events["senders"][idc_times]
    times_subset = events["times"][idc_times]
    tmp = times_subset
    tmp = np.sort(tmp,axis=0)
    time, rate = np.unique(tmp,return_counts=True)
    rate = rate / np.max(rate)

    ax.plot(time,rate)
    ax.set_ylabel(ylabel)
    #margin = np.abs(np.max(senders_subset) - np.min(senders_subset)) * 0.1
    #ax.set_ylim(np.min(senders_subset) - margin, np.max(senders_subset) + margin)

def plot_spike_delta(ax, events, ylabel, xlims):
    idc_times = (events["times"] > xlims[0]) & (events["times"] < xlims[1])
    times_subset = events["times"][idc_times]
    senders_subset = events["senders"][idc_times]

    tmp = np.column_stack([times_subset, senders_subset])
    tmp = tmp[tmp[:, 0].argsort()]
    tmp = tmp.T
    times_subset = tmp[0]
    senders_subset = tmp[1]

    time = np.arange(xlims[0]+1,xlims[1]-1,1)
    f = np.arange(xlims[0]+1,xlims[1]-1,1,dtype=np.float64)
    f[0] = objective_func(1)[0][0]
    pos = np.logspace(np.log10(model.fmin),np.log10(model.fmax),model.channels)
    neg = -pos 
    thresholds = np.empty(2*model.channels)
    thresholds[0::2] = pos
    thresholds[1::2] = neg
    for t in range(len(f)-1):
        activations = 0
        t_ = np.argwhere(times_subset==time[t])
        nrns = senders_subset[t_]
        if len(nrns) > 0:
            active = np.max(nrns)
            for nrn, threshold in zip(nrns_in, thresholds):
                if nrn == active:
                    activations = threshold
        f[t+1] = f[t] + activations
    ax.plot(time, f)
        

for title, xlims, savename in zip(
    ["Dynamic variables before training", "Dynamic variables after training"],
    [
        (0, model.steps["sequence"]),
        #(0, 100),
        #(int(((model.n_iter_sim) * model.group_size - 0.55)* model.steps["sequence"]), int((model.n_iter_sim * model.group_size - 0.45) * model.steps["sequence"])),
        (((model.n_iter_sim) * model.group_size - 1)* model.steps["sequence"], model.n_iter_sim * model.group_size * model.steps["sequence"]),
    ],["before","after"]
):
    
        fig, axs = plt.subplots(7, 1, sharex=True, figsize=(8, 14), gridspec_kw={"hspace": 0.4, "left": 0.2})
        fig.suptitle(title)

        #axs[0].plot(np.arange(xlims[0],xlims[1],1),objective_func(xlims[1]-xlims[0]).squeeze())
        plot_recordable(axs[0], events_mm_out, "target_signal", r"$y^*_k$" + "\n", xlims,overlay=True)
        axs[0].set_ylabel("x(t) \n")
        #plot_recordable(axs[0], events_mm_out, "target_signal", r"$y^*_k$" + "\n", xlims)
        #plot_spike_rate(axs[0], events_sr_in, r"$f(x)$" + "\n", xlims)
        plot_spike_delta(axs[0], events_sr_in, r"$f(x)$" + "\n", xlims)

        plot_spikes(axs[1], events_sr_in, r"$z_i$" + "\n", xlims)
        plot_spikes(axs[2], events_sr_rec, r"$z_j$" + "\n", xlims)

        plot_recordable(axs[3], events_mm_rec, "V_m", r"$v_j$" + "\n(mV)", xlims)
        if model.adaptive:
            plot_recordable(axs[3], events_mm_rec, "V_th_adapt", r"$v_j$" + "\n(mV)", xlims,overlay=True)
        #axs[2].set_ylim(0,15)
        plot_recordable(axs[4], events_mm_rec, "surrogate_gradient", r"$\psi_j$" + "\n", xlims)

        plot_recordable(axs[5], events_mm_out, "V_m", r"$v_k$" + "\n(mV)", xlims)
        plot_recordable(axs[5], events_mm_out, "target_signal", r"$y^*_k$" + "\n", xlims,overlay=True)

        plot_recordable(axs[6], events_mm_rec, "learning_signal", r"$L_j$" + "\n(pA)", xlims)
        if savename == 'after':
            mid = int(((model.n_iter_sim) * model.group_size - 0.5)* model.steps["sequence"])
            for i in range(7):
                axs[i].axvline(mid+4,c='k',linewidth=0.5,linestyle='--')
                #axs[i].axvline(mid+23,c='k',linewidth=0.5,linestyle='--')
                
        #plot_recordable(axs[7], events_mm_out, "readout_signal", r"$y_k$" + "\n", xlims,overlay=True)
        #plot_recordable(axs[8], events_mm_out, "readout_signal", r"$y_k$" + "\n", xlims)
        #plot_recordable(axs[9], events_mm_out, "error_signal", r"$y_k-y^*_k$" + "\n", xlims)

        axs[-1].set_xlabel(r"$t$ (ms)")
        axs[-1].set_xlim(*xlims)

        fig.align_ylabels()

        #fig.savefig(f"{plots_dir}{timestamp}/recorders_{savename}.eps")
#%% weight evolution

def plot_weight_time_course(ax, events, nrns, label, ylabel):
    sender_label, target_label = label.split("_")
    nrns_senders = nrns[sender_label]
    nrns_targets = nrns[target_label]

    for sender in np.unique(events["senders"]):
        for target in np.unique(events["targets"]):
            if sender in nrns_senders and target in nrns_targets:
                idc_syn = (events["senders"] == sender) & (events["targets"] == target)
                idc_syn_pre = (weights_pre_train[label]["source"] == sender) & (
                    weights_pre_train[label]["target"] == target
                )

                if np.any(idc_syn) and np.any(idc_syn_pre):
                    times = np.concatenate([[0.0], events["times"][idc_syn]])

                    weights = np.concatenate(
                        [np.array(weights_pre_train[label]["weight"])[idc_syn_pre], events["weights"][idc_syn]]
                    )
                    ax.step(times, weights, c=colors["blue"])
        ax.set_ylabel(ylabel)
        #ax.set_ylim(np.min(tot_weights)*1.1, np.max(tot_weights)*1.1)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3, 4))
fig.suptitle("Weight time courses")

nrns = {
    "in": model.nrns_in.tolist(),
    "rec": model.nrns_rec.tolist(),
    "out": model.nrns_out.tolist(),
}

plot_weight_time_course(axs[0], events_wr, nrns, "in_rec", r"$W_\text{in}$ (pA)")
plot_weight_time_course(axs[1], events_wr, nrns, "rec_rec", r"$W_\text{rec}$ (pA)")
plot_weight_time_course(axs[2], events_wr, nrns, "rec_out", r"$W_\text{out}$ (pA)")

axs[-1].set_xlabel(r"$t$ (ms)")
axs[-1].set_xlim(0, model.duration["task"])

fig.align_ylabels()
fig.tight_layout()

#%% weight matrices
import matplotlib as mpl
fig, axs = plt.subplots(3, 2, sharex="col", sharey="row")
fig.suptitle("Weight matrices")

all_w_extrema = []
all_tails = []

for k in weights_pre_train.keys():
    w_pre = weights_pre_train[k]["weight"]
    w_post = weights_post_train[k]["weight"]
    all_tails.append([np.percentile(np.abs(w_pre),95),np.percentile(np.abs(w_post),95)])
    all_w_extrema.append([np.min(w_pre), np.max(w_pre), np.min(w_post), np.max(w_post)])

vmax = np.max([np.abs(np.min(all_w_extrema)),np.min(all_w_extrema)])
v_tail = np.mean(all_tails)

dist = v_tail/(2*vmax)


'''cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["green"]),(0.5-dist, colors["blue"]), (0.5, colors["white"]), (0.5+dist, colors["red"]),(1, colors["purple"]))
)'''
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "cmap", ((0.0, colors["blue"]), (0.5, colors["white"]), (1, colors["red"]))
)


args = {"cmap": cmap, "vmin": -vmax, "vmax": vmax}

for i, weights in zip([0, 1], [weights_pre_train, weights_post_train]):
    axs[0, i].pcolormesh(weights["in_rec"]["weight_matrix"].T, **args)
    axs[1, i].pcolormesh(weights["rec_rec"]["weight_matrix"], **args)
    cmesh = axs[2, i].pcolormesh(weights["rec_out"]["weight_matrix"], **args)

    axs[2, i].set_xlabel("recurrent\nneurons")

axs[0, 0].set_ylabel("input\nneurons")
axs[1, 0].set_ylabel("recurrent\nneurons")
axs[2, 0].set_ylabel("readout\nneurons")
fig.align_ylabels(axs[:, 0])

axs[0, 0].text(0.5, 1.1, "before training", transform=axs[0, 0].transAxes, ha="center")
axs[0, 1].text(0.5, 1.1, "after training", transform=axs[0, 1].transAxes, ha="center")

axs[2, 0].yaxis.get_major_locator().set_params(integer=True)

fig.tight_layout()

cax = fig.add_axes([0.98,0.1,0.025,0.8])
cbar = plt.colorbar(cmesh,cax=cax, label="weight (pA)")
cbar.set_ticks([-vmax,-v_tail,0,v_tail,vmax])

#fig.savefig(f"{plots_dir}{timestamp}/weights_before_after.eps")
plt.show()
# %%
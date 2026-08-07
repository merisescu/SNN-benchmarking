#%% imports

import numpy as np
import setup_api, dataset_api

#%%

# load MNIST batch loader and initialise dataset
data_loader = dataset_api.MNIST_loader()
n_in, n_out = data_loader.setup_MNIST()


#%% set parameters

# simulation parameters
steps = {
    "sequence": 300,  # time steps of one full sequence
    "learning_window": 10,  # time steps of window with non-zero learning signals
}

# neuron parameters
eprop_isi_trace_cutoff = 100 # after this many steps after a spike, stops computing eligibility trace
params_nrn_out = {
    "tau_m": 100,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff
}

params_nrn_rec = {
    "tau_m": 30,
    "kappa": 0.99,
    "c_reg": 2,
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff
}

data_loader.setup_MNIST(activity_threshold=0.02,val_split=0.1) # manually delete pixel_activations.csv to reinitialise
model = setup_api.eprop_model(data_loader)
model.set_timing(1.0,1,steps)
model.input_layer(n_in)
model.recurrent_layer(150,params_nrn_rec)
model.output_layer(n_out,params_nrn_out)
model.recorders()
model.connect(optimizer='gradient_descent',)
model.compile()
model.fit(n_iter_train=200,group_size=100) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

#results = model.get_results()

#accuracy = 1-results['error']
#spikes = results['spikes']
# %%

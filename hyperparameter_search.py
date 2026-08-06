import optuna
import numpy as np

import setup_api
import dataset_api

#%% NMNIST hyperparameter search

active_pixels = dataset_api.setup_MNIST()
n_in = active_pixels.size

# using standard LIF
def LIF_objective(trial):

    # neuron parameters
    params_rec_nrn = {
        "tau_m": trial.suggest_float('leakage')
    }
    data_loader = dataset_api.MNIST_loader()
    data_loader.setup_MNIST(activity_threshold=0.02,val_split=0.1)
    model = setup_api.eprop_model(data_loader)
    model.input_layer(n_in)
    model.recurrent_layer(trial.suggest_int('reccurent_neurons',50,200,step=25),params_rec_nrn)
    model.output_layer()
    model.compile()
    model.fit(profiling='sparse')

    results = model.get_results()

    accuracy = 1-results['error']
    spikes = results['spikes']

    return spikes, accuracy

LIF_study = optuna.create_study(directions=['minimize','maximize'],sampler=optuna.samplers.GridSampler())
LIF_study.optimize(LIF_objective,n_trials=30)

# using adaptive LIF

#def ALIF_objective(trial):

#%% Spiking speech commands hyperparameter search
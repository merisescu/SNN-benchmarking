#%% import
import optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
import numpy as np

import setup_api
import dataset_api

#%% NMNIST hyperparameter search

# load MNIST batch loader and initialise dataset
data_loader = dataset_api.MNIST_loader()
n_in, n_out = data_loader.setup_MNIST()
#%%
# using standard LIF
def LIF_objective(trial):

    # neuron parameters
    # TODO: add all necessary hyperparameters
    tau_m = trial.suggest_float('leakage') #leakage time constant
    

    params_nrn_out = {
        "tau_m": tau_m
    }

    params_nrn_rec = {
        "tau_m": tau_m
    }

    data_loader.setup_MNIST(activity_threshold=0.02,val_split=0.1) # manually delete pixel_activations.csv to reinitialise
    model = setup_api.eprop_model(data_loader)
    model.input_layer(n_in)
    model.recurrent_layer(trial.suggest_int('reccurent_neurons',50,200,step=25),params_nrn_rec)
    model.output_layer(n_out,params_nrn_out)
    model.compile(profiling='sparse')
    model.fit(n_iter_train=200,group_size=100) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

    results = model.get_results()

    accuracy = 1-results['error']
    spikes = results['spikes']

    return spikes, accuracy

def run_optimization(_):
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./data/hyperparameters/journal.log")),
        load_if_exists=True, # Useful for multi-process or multi-node optimization.
    )
    study.optimize(LIF_objective, n_trials=30)

#LIF_study = optuna.create_study(directions=['minimize','maximize'],sampler=optuna.samplers.TPESampler())
#LIF_study.optimize(LIF_objective,n_trials=30)

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(run_optimization, range(12))

# using adaptive LIF

#def ALIF_objective(trial):

#%% Spiking speech commands hyperparameter search
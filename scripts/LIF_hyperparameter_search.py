#%% import
import optuna
import os
import numpy as np
import scripts.api.setup as setup
import scripts.api.dataset as dataset

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']

#%% NMNIST hyperparameter search

# load MNIST batch loader and initialise dataset
data_loader = dataset.MNIST_loader()
n_in, n_out = data_loader.setup_MNIST()
#%%

no_trials = 2

# using standard LIF
def objective(trial):
    accuracy = np.zeros(no_trials)
    spikes = np.zeros(no_trials)

    # simulation parameters

    steps = {
    "sequence": 300,  # time steps of one full sequence
    "learning_window": 10,  # time steps of window with non-zero learning signals
    }

    # neuron parameters
    # TODO: add all necessary hyperparameters
    eprop_isi_trace_cutoff = 100
    params_nrn_out = {
    "tau_m": trial.suggest_float('tau_m_out', 1, 200), #leakage time constant
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "V_m": 0.,
    "E_L": 0.
    }

    params_nrn_rec = {
    "tau_m": trial.suggest_float('tau_m_rec', 1, 200), #leakage time constant
    "c_reg": trial.suggest_float('c_reg', 0, 5),
    "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
    "E_L": 0.,
    "V_m": 0.,
    "V_th": trial.suggest_float('V_th', 0.1, 1),
    "t_ref": trial.suggest_float('t_ref', 0, 20),
    "kappa": trial.suggest_float('kappa',0.85,1),
    "kappa_reg": trial.suggest_float('kappa_reg',0.85,1),
    "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
    "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
    "surrogate_gradient_width": 1.0 / 1.7,  # width scaling of the pseudo-derivative

    "f_target": trial.suggest_float("f_target", 1, 60), # Hz
    }

    for i in range(no_trials):

        data_loader.setup_MNIST(activity_threshold=0.02,val_split=0.1) # manually delete pixel_activations.csv to reinitialise
        model = setup.eprop_model(num_threads=1,num_virtual_procs=4)
        model.timing(1,1,steps)
        model.input_layer(n_in)
        model.recurrent_layer(trial.suggest_int('reccurent_neurons',50,200,step=25),params_nrn_rec,adaptive=False)
        model.output_layer(n_out,params_nrn_out)
        model.recorders(profiling='sparse',save_to_disk=False)
        model.connect(sparsity=[0.75,0.99,0],eta_train=5e-3,optimizer='gradient_descent')
        model.compile() # if you want more data add custom metrics, and any needed recorder information from the setup api. full recording is not suitable for hyperparameter search
        model.fit(data_loader,n_iter_train=100,group_size=128) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

        if model.pruned:
            raise optuna.TrialPruned()

        results, _ = model.get_results()

        val_idx = np.asarray(results['label'])
        val_idx = np.argwhere(val_idx == 'validation')

        accuracy[i] = np.mean(1-results['error'][val_idx])
        spikes[i] = np.mean(results['spikes'][val_idx])

    accuracy = np.mean(accuracy)
    spikes = np.mean(spikes)

    return spikes, accuracy

if __name__ == "__main__":
    study = optuna.load_study(
        study_name="NMNIST_LIF_v2",
        storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}",
    )
    study.optimize(objective, n_trials=None)
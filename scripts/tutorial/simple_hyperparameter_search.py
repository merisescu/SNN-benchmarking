#%% import
import optuna
import os

import scripts.api.setup as setup
import scripts.api.dataset as dataset

#%% NMNIST hyperparameter search

# load MNIST batch loader and initialise dataset
data_loader = dataset.MNIST_loader()
n_in, n_out = data_loader.setup_MNIST()
#%%
# using standard LIF
def LIF_objective(trial):
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
        "kappa": trial.suggest_float('kappa',0.8,1),
        "kappa_reg": trial.suggest_float('kappa_reg',0.8,1),
        "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
        "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
        "surrogate_gradient_width": 1.0 / 1.7,  # width scaling of the pseudo-derivative
    
        #"adapt_beta": trial.suggest_float('adapt_beta',0,2),
        #"adapt_tau": trial.suggest_float('adapt_tau',1,200),
    
        "f_target": trial.suggest_float("f_target", 1, 60), # Hz
    }

    data_loader.setup_MNIST(activity_threshold=0.02,val_split=0.1) # manually delete pixel_activations.csv to reinitialise
    model = setup.eprop_model(num_threads=1,num_virtual_procs=4)
    model.timing(1.0,1,steps)
    model.input_layer(n_in)
    model.recurrent_layer(trial.suggest_int('reccurent_neurons',50,200,step=25),params_nrn_rec,adaptive=False)
    model.output_layer(n_out,params_nrn_out)
    model.recorders(profiling='sparse',save_to_disk=False)
    model.connect(sparsity=[0.9,0.99,0],eta_train=5e-3,optimizer='gradient_descent')
    model.compile() # if you want more data add custom metrics, and any needed recorder information from the setup api. full recording is not suitable for hyperparameter search
    model.fit(data_loader,n_iter_train=200,group_size=100) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

    results = model.get_results()

    accuracy = 1-results['error'][-1]
    spikes = results['spikes'][-1]

    return spikes, accuracy

LIF_study = optuna.create_study(storage=optuna.storages.InMemoryStorage(),directions=['minimize','maximize'],sampler=optuna.samplers.TPESampler())
LIF_study.optimize(LIF_objective,n_trials=50)

#%%

optuna.visualization.plot_pareto_front(LIF_study,target_names=['spikes','accuracy'])

# %%
optuna.visualization.plot_slice(LIF_study, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','reccurent_neurons'], target=lambda t: t.values[1],target_name="accuracy")
# %%
optuna.visualization.plot_slice(LIF_study, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','reccurent_neurons'], target=lambda t: t.values[0],target_name="spikes")
# %%


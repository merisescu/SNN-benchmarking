#%% imports

import numpy as np
import scripts.api.setup as setup
import scripts.api.dataset as  dataset
import optuna
import os
import pandas as pd
import matplotlib.pyplot as plt

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']
data_dir = "./data/"
plots_dir = "./figures/"

nmnist_lif = optuna.load_study(study_name="NMNIST_LIF_v2",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

nmnist_alif = optuna.load_study(study_name="NMNIST_ALIF_v2",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

studies = [nmnist_lif,nmnist_alif]

all_hyperparams = []
for study in studies:
    all_hyperparams.append([trial.params for trial in study.best_trials])

#%% set up trials
def trial(hyperparams,adaptive):
    data_loader = dataset.MNIST_loader()
    n_in, n_out = data_loader.setup_MNIST()

    steps = {
        "sequence": 300,  # time steps of one full sequence
        "learning_window": 10,  # time steps of window with non-zero learning signals
        }
    
    eprop_isi_trace_cutoff = 100
    params_nrn_out = {
        "tau_m": hyperparams['tau_m_out'], #leakage time constant
        "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
        "V_m": 0.,
        "E_L": 0.
    }

    params_nrn_rec = {
        "tau_m": hyperparams['tau_m_rec'], #leakage time constant
        "c_reg": hyperparams['c_reg'],
        "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
        "E_L": 0.,
        "V_m": 0.,
        "V_th": hyperparams['V_th'],
        "t_ref": hyperparams['t_ref'],
        "kappa": hyperparams['kappa'],
        "kappa_reg": hyperparams['kappa_reg'],
        "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
        "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
        "surrogate_gradient_width": 1.0 / 1.7,  # width scaling of the pseudo-derivative

        "f_target": hyperparams['f_target'], # Hz
    }

    if adaptive:
        params_nrn_rec.update({
            "adapt_beta": hyperparams['adapt_beta'],
            "adapt_tau": hyperparams['adapt_tau']
            }
        )

    data_loader.setup_MNIST(activity_threshold=0.02,val_split=0.1) # manually delete pixel_activations.csv to reinitialise
    model = setup.eprop_model(num_threads=1,num_virtual_procs=4)
    model.timing(1,1,steps)
    model.input_layer(n_in)
    model.recurrent_layer(hyperparams['reccurent_neurons'],params_nrn_rec,adaptive=adaptive)
    model.output_layer(n_out,params_nrn_out)
    model.recorders(profiling='sparse',save_to_disk=False)
    model.connect(sparsity=[0.75,0.99,0],eta_train=5e-3,optimizer='gradient_descent')
    weights_pre_train = model.compile()
    model.fit(data_loader,n_iter_train=200,group_size=128,n_validations=10) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

    results, events = model.get_results()

    return results, events

#%%

temp = []
studies_, indexes = enumerate(studies)
for study, i, adaptive in zip(studies_, indexes, [False, True]):
    for hyperparams in all_hyperparams[i]:
        results, _ = trial(hyperparams, adaptive)
        accuracy = 1-results['error']
        spikes = results['spikes']
        auc = results['auc']
        temp.append({
            'adaptive': np.full(len(accuracy), adaptive, dtype=bool),
            'accuracy': accuracy,
            'spikes': spikes,
            'auc': auc
            }
        )
studies_df = pd.DataFrame(temp)
lif_nmnist_df = studies_df[studies_df['adaptive'==False]]
alif_nmnist_df = studies_df[studies_df['adaptive'==True]]

dfs = [lif_nmnist_df,alif_nmnist_df]

for df, title in zip(dfs, ['LIF', 'ALIF']):
    df = df.sort_values(by="spikes")
    plt.scatter(df['spikes'],df['accuracy'],label=title)


# %%

#%% imports

import optuna
import optunahub
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']
data_dir = "./data/"
plots_dir = "./figures/"

#%% LIF
nmnist_lif = optuna.load_study(study_name="NMNIST_LIF_sparse",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(nmnist_lif,target_names=['spikes','accuracy'])

#%%
optuna.visualization.plot_hypervolume_history(nmnist_lif,[1000,0.8])

#%%
optuna.visualization.plot_hypervolume_history(nmnist_lif,[2000,0.8])


#%%
optuna.visualization.plot_slice(nmnist_lif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','recurrent_neurons'], target=lambda t: t.values[1],target_name="accuracy")
#%%
optuna.visualization.plot_slice(nmnist_lif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','recurrent_neurons'], target=lambda t: t.values[0],target_name="spikes")
#%% ALIF
nmnist_alif = optuna.load_study(study_name="NMNIST_ALIF_sparse",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(nmnist_alif,target_names=['spikes','accuracy'])

#%%
optuna.visualization.plot_hypervolume_history(nmnist_alif,[1000,0.8])

#%%
optuna.visualization.plot_hypervolume_history(nmnist_alif,[2000,0.8])


#%%
optuna.visualization.plot_slice(nmnist_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','recurrent_neurons','adapt_beta','adapt_tau'], target=lambda t: t.values[1],target_name="accuracy")
# %%
optuna.visualization.plot_slice(nmnist_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','recurrent_neurons','adapt_beta','adapt_tau'], target=lambda t: t.values[0],target_name="spikes")
# %% custom plots

# Pareto fronts of MNIST studies
studies = [nmnist_lif, nmnist_alif]
xmin = 500
xmax = 500
ymin = 0.85
ymax = 0.85
for study, study_name in zip(studies,['LIF','ALIF']):
    optimal_df = pd.DataFrame([
        {
            "spikes": trial.values[0],
            "accuracy": trial.values[1],
            **trial.params,
        }
        for trial in study.best_trials
    ])
    trials_df = pd.DataFrame([
            {
                "spikes": trial.values[0],
                "accuracy": trial.values[1],
                **trial.params,
            }
            for trial in study.trials
            if trial.values is not None
        ])
    optimal_df = optimal_df.sort_values(by=["spikes"])
    plt.step(optimal_df['spikes'],optimal_df['accuracy'],where='post',label=study_name)
    plt.scatter(trials_df['spikes'],trials_df['accuracy'],s=10)
    xmin = np.min([xmin,optimal_df.min(axis=0)['spikes']])
    xmax = np.max([xmax,optimal_df.max(axis=0)['spikes']])
    ymin = np.min([ymin,optimal_df.min(axis=0)['accuracy']])
    ymax = np.max([ymax,optimal_df.max(axis=0)['accuracy']])
xmargin = 0.05 * (xmax-xmin)
ymargin = 0.05 * (ymax-ymin)
plt.xlabel('spikes per inference')
plt.ylabel('accuracy')
plt.xlim([xmin-xmargin,xmax+xmargin])
plt.ylim([ymin-ymargin,ymax+ymargin])
plt.legend()
plt.savefig(f"{plots_dir}nmnist_pareto_fronts.eps")

# %%
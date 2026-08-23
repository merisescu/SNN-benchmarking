#%% imports

import optuna
import os

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']

#%% LIF
nmnist_lif = optuna.load_study(study_name="NMNIST_LIF_v2",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(nmnist_lif,target_names=['spikes','accuracy'])
#%%
optuna.visualization.plot_slice(nmnist_lif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','reccurent_neurons'], target=lambda t: t.values[1],target_name="accuracy")
#%%
optuna.visualization.plot_slice(nmnist_lif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','reccurent_neurons'], target=lambda t: t.values[0],target_name="spikes")
#%% ALIF
nmnist_alif = optuna.load_study(study_name="NMNIST_ALIF_v2",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(nmnist_alif,target_names=['spikes','accuracy'])
#%%
optuna.visualization.plot_slice(nmnist_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','reccurent_neurons','adapt_beta','adapt_tau'], target=lambda t: t.values[1],target_name="accuracy")
# %%
optuna.visualization.plot_slice(nmnist_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','reccurent_neurons','adapt_beta','adapt_tau'], target=lambda t: t.values[0],target_name="spikes")
# %%

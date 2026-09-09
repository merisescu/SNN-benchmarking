#%% imports

import optuna
from optuna._hypervolume import compute_hypervolume
import optunahub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'

import seaborn as sns
import cnsplots as cns
cns.settings.palette_qual = 'Cell'
cns.settings.legend_markerscale = 1.3
cns.settings.savefig_dpi = 400
cns.settings.axes_spines_top = False
cns.settings.axes_spines_right = False

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']
data_dir = "./data/"
plots_dir = "./figures/"

#%% functions

def get_hypervolume(study,bounds):
    # fix for maximisation goal
    signs = np.asarray([
    1 if d == optuna.study.StudyDirection.MINIMIZE else -1
    for d in study.directions
    ])
    pareto_front = np.array([
    trial.values
    for trial in study.best_trials
    ])
    pareto_front = pareto_front * signs
    bounds_ = bounds * signs
    idxs = np.all(pareto_front < bounds_, axis = 1)
    pareto_front = pareto_front[idxs]

    V = compute_hypervolume(pareto_front,np.array(bounds_))
    print(V)
    return V

def pareto_plot(studies,titles,xlabel,ylabel,savetitle,bounds=[0,1000,0,1],hypervolume_bounds=[2000,0.8],xscale=1,use_error=False):
    optimal_dfs = {}
    trials_dfs = {}

    xmin,xmax,ymin,ymax = bounds

    for study, study_name in zip(studies,titles):
        optimal_dfs.update({study_name:pd.DataFrame([
            {
                "spikes": trial.values[0]*xscale,
                "accuracy": trial.values[1],
                **trial.params,
            }
            for trial in study.best_trials
        ])})
        trials_dfs.update({study_name:pd.DataFrame([
                {
                    "spikes": trial.values[0]*xscale,
                    "accuracy": trial.values[1],
                    **trial.params,
                }
                for trial in study.trials
                if trial.values is not None
            ])})
        optimal_dfs[study_name] = optimal_dfs[study_name].sort_values(by=["spikes"])
        optimal_dfs[study_name]["errors"] = 1-optimal_dfs[study_name]['accuracy']
        trials_dfs[study_name]["errors"] = 1-trials_dfs[study_name]['accuracy']

        optimal_dfs[study_name] = optimal_dfs[study_name].assign(neuron=study_name)
        trials_dfs[study_name] = trials_dfs[study_name].assign(neuron=study_name)

        xmin = np.min([xmin,optimal_dfs[study_name].min(axis=0)['spikes']])
        xmax = np.max([xmax,optimal_dfs[study_name].max(axis=0)['spikes']])
        if use_error:
            ymin_ = np.min([ymin,optimal_dfs[study_name].max(axis=0)['errors']])
            ymax_ = np.max([ymax,optimal_dfs[study_name].min(axis=0)['errors']])
        else:
            ymin_ = np.min([ymin,optimal_dfs[study_name].min(axis=0)['accuracy']])
            ymax_ = np.max([ymax,optimal_dfs[study_name].max(axis=0)['accuracy']])

    hvs = []
    for study in studies:
        hvs.append(get_hypervolume(study,hypervolume_bounds))
    df_order = np.argsort(hvs)[::-1]

    #optimal_df = pd.concat([study for study in reversed(optimal_dfs.values())])
    #trials_df = pd.concat([study for study in reversed(trials_dfs.values())])

    optimal_df = pd.concat([optimal_dfs[titles[i]] for i in df_order])
    trials_df = pd.concat([trials_dfs[titles[i]] for i in df_order])

    trials_df = trials_df[trials_df['spikes']< xmax]
    if use_error:
        y_ = "errors"
    else:
        y_ = "accuracy"
    
    trials_df = trials_df[trials_df[y_] < ymax]
    trials_df = trials_df[trials_df[y_] > ymin]

    cns.figure(250,340)
    ax = plt.gca()

    sns.lineplot(data=optimal_df,x='spikes',y=y_,drawstyle='steps-post',hue=optimal_df["neuron"],hue_order=titles,legend=False,ax=ax)
    sns.scatterplot(data=trials_df,x='spikes',y=y_,s=12,alpha=0.4,hue=trials_df["neuron"],hue_order=titles,ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    for i,collection in enumerate(ax.collections):
        collection.set_clip_on(False)
    cns.savefig(f"{plots_dir}{savetitle}.pdf")

#%% LIF
nmnist_lif = optuna.load_study(study_name="NMNIST_LIF_sparse",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(nmnist_lif,target_names=['spikes','accuracy'])

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
optuna.visualization.plot_hypervolume_history(nmnist_alif,[2000,0.8])

#%%
optuna.visualization.plot_slice(nmnist_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','recurrent_neurons','adapt_beta','adapt_tau'], target=lambda t: t.values[1],target_name="accuracy")
# %%
optuna.visualization.plot_slice(nmnist_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','recurrent_neurons','adapt_beta','adapt_tau'], target=lambda t: t.values[0],target_name="spikes")
# %% custom plots

# Pareto fronts of MNIST studies
studies = [nmnist_lif, nmnist_alif]
titles = ["LIF","ALIF"]
pareto_plot(studies,titles,"Spikes per inference","Classification accuracy",savetitle="nmnist_pareto_fronts",bounds=[50,2000,0.75,0.95])

#%% LIF Function regression
mg_lif = optuna.load_study(study_name="MG_LIF",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(mg_lif,target_names=['spikes','accuracy'])

#%%
optuna.visualization.plot_hypervolume_history(mg_lif,[150000,0.9])

#%% ALIF Function regression
mg_alif = optuna.load_study(study_name="MG_ALIF",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(mg_alif,target_names=['spikes','accuracy'])

#%%
optuna.visualization.plot_hypervolume_history(mg_alif,[150000,0.9])

#%%
optuna.visualization.plot_slice(mg_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','adapt_beta','adapt_tau','in_channels','recurrent_neurons'], target=lambda t: t.values[1],target_name="accuracy")
# %%
optuna.visualization.plot_slice(mg_alif, params=['tau_m_rec','tau_m_out','V_th','t_ref','f_target','kappa','kappa_reg','c_reg','adapt_beta','adapt_tau','in_channels','recurrent_neurons'], target=lambda t: t.values[0],target_name="spikes")

#%% Izhikevich Function regression
mg_izh = optuna.load_study(study_name="MG_Izh",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}")

optuna.visualization.plot_pareto_front(mg_izh,target_names=['spikes','accuracy'])

#%%
optuna.visualization.plot_hypervolume_history(mg_izh,[150000,0.9])

#%%
optuna.visualization.plot_slice(mg_izh, params=["a","b","c","d","c_reg","kappa","kappa_reg","tau_m_out","in_channels","recurrent_neurons"], target=lambda t: t.values[1],target_name="accuracy")
# %%
optuna.visualization.plot_slice(mg_izh, params=["a","b","c","d","c_reg","kappa","kappa_reg","tau_m_out","in_channels","recurrent_neurons"], target=lambda t: t.values[0],target_name="spikes")

# %%

studies = [mg_lif,mg_alif,mg_izh]
titles = ["LIF","ALIF","Izhikevich"]
n_L = 1500/197
pareto_plot(studies,titles,'Spikes per $L$','One-step error',savetitle="mg_pareto_fronts",bounds=[0,1780,0,0.12],hypervolume_bounds=[150000,0.9],xscale=1/n_L,use_error=True) # scale from one sequence (20L) to (L)

# %%

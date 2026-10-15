#%% import
import optuna
import os
import numpy as np
import scripts.api.setup as setup
from reservoirpy.datasets import mackey_glass

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
db = os.environ['POSTGRES_DB']

#%% func definition

# Using 1D Mackey-Glass
func_max = 1.32
# with standard parameters
lyapunov_time = 197
resolution = lyapunov_time / 75
n_timesteps = round(20 * lyapunov_time / resolution) # (20 * 75)
#n_timesteps = 3000
# keeping for clarity of calculation
'''def mg_func(steps):
    f = mackey_glass(steps).T
    f /= func_max
    return f'''

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

        dx = (
            beta * delayed / (1.0 + delayed**n)
            - gamma * x[t]
        )

        x[t + 1] = x[t] + dt * dx
    return x[delay_steps+1:]/func_max

def mg_func(steps):
    return [mackey_glass(n_timesteps=steps,dt=resolution)]

#%%

no_trials = 2

# using standard LIF
def objective(trial):
    accuracy = np.zeros(no_trials)
    spikes = np.zeros(no_trials)

    # simulation parameters

    steps = {
    "sequence": n_timesteps,  # time steps of one full sequence
    }

    # neuron parameters
    # TODO: add all necessary hyperparameters
    eprop_isi_trace_cutoff = 300
    params_nrn_out = {
        "tau_m": trial.suggest_float('tau_m_out', 1, 300), #leakage time constant
        "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
        "V_m": 0.,
        "E_L": 0.
    }

    params_nrn_rec = {
        "tau_m": trial.suggest_float('tau_m_rec', 1, 300), #leakage time constant
        "c_reg": trial.suggest_float('c_reg', 0, 20),
        "eprop_isi_trace_cutoff": eprop_isi_trace_cutoff,
        "E_L": 0.,
        "V_m": 0.,
        "V_th": trial.suggest_float('V_th', 0.1, 2),
        "t_ref": trial.suggest_float('t_ref', 0, 30),
        "kappa": trial.suggest_float('kappa',0.8,1),
        "kappa_reg": trial.suggest_float('kappa_reg',0.8,1),
        "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
        "surrogate_gradient_height": 0.5,  # height scaling of the pseudo-derivative
        "surrogate_gradient_width": 1.0 / 1.7,  # width scaling of the pseudo-derivative

        "adapt_beta": trial.suggest_float('adapt_beta',0,3),
        "adapt_tau": trial.suggest_float('adapt_tau',1,300),

        "f_target": trial.suggest_float("f_target", 0.1, 20), # Hz
    }

    n_rec = trial.suggest_int('recurrent_neurons',50,300,step=25)

    for i in range(no_trials):
        model = setup.eprop_model(mode='Regressor',num_threads=1,num_virtual_procs=4)
        model.timing(1,1,steps)
        model.input_layer(1,encoding='delta',channels=trial.suggest_int("in_channels", 1, 16))
        model.recurrent_layer(n_rec,params_nrn_rec,adaptive=True)
        model.output_layer(1,params_nrn_out)
        model.recorders(profiling='sparse',save_to_disk=False)
        model.connect(sparsity=[0.8,0.8,0],eta_train=5e-3,optimizer='gradient_descent')
        model.compile() # if you want more data add custom metrics, and any needed recorder information from the setup api. full recording is not suitable for hyperparameter search
        model.fit(function=mg_func,n_iter_train=250,group_size=1) # n_iter_train = number of training iterations if not using early_stop, group size = how many items to run in sequence per training phase

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
        study_name="MG_ALIF_dt",
        storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{db}",
    )
    study.optimize(objective, n_trials=None)
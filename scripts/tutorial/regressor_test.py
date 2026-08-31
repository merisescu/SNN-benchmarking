#%% imports
import scripts.api.setup as setup
from reservoirpy.datasets import mackey_glass
import numpy as np

#%% function defs
# wrapper func to pass to training API
# normalise your input function
# pass returns with dimension (num_timesteps, n)

# Using 1D Mackey-Glass
func_max = 1.32
# with standard parameters
lyapunov_time = 197
resolution = lyapunov_time / 75
n_timesteps = round(20 * lyapunov_time / resolution) # (20 * 75)
# keeping for clarity of calculation
def objective_func(steps):
    return mackey_glass(steps,h=resolution).T/func_max


#%%

steps = {"sequence": n_timesteps}
test_model = setup.eprop_model(mode='Regressor')
test_model.timing(1,1,steps)
test_model.function = objective_func
test_model.regressor_coding()

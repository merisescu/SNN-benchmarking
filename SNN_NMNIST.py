#%% Imports

# utils

# maths
import numpy as np

#network
import nest

# visualisation
import cnsplots as cns

#%% Import and format data

#%% set variables for development

#%% Extra classes and functions

#%% initialise NEST

params_setup = {
    "print_time": False,  # if True, print time progress bar during simulation, set False if run as code cell
    "resolution": 1.0,
    "total_num_virtual_procs": 4,  # number of virtual processes, set in case of distributed computing
}

nest.ResetKernel()
nest.set(**params_setup)
nest.verbosity = nest.VerbosityLevel.FATAL


#%% Initialise neurons

#%% Initialise connections (synapses)

#%% Initialise recorders

#%% Evaluate initial model

#%% Simulate

#%% Extract simulation data

#%% Visualise
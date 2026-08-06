import numpy as np
import nest
import time
#from sklearn.metrics import 
#%%

# we always want trainer to receive data as a list len(n_in) where each list item is an array that contains spike times for that neuron
# in order to allow for use with any dataset, we use a hook which should be defined outside the class to load each batch
class Trainer:
    def __init__(self, batch_loader):
        self.batch_loader = batch_loader
        self.results_dict = {
            "error": [],
            "loss": [],
            "iteration": [],
            "label": [],
        }
        self.n_iter_sim = 0
        self.phase_label_previous = ""
        self.error = 0
        self.k_iter = 0
        self.early_stop = False
        self.profiling =='sparse'

    def fit(self,n_iter_train):
        while self.k_iter < n_iter_train and not self.early_stop:
            self.run_validation()
            self.run_early_stopping()
            self.run_training()
            self.k_iter += 1
            print('epoch: ', self.k_iter)
        return

    def run_training(self):
        self.run_phase()

    def run_phase(self):
        params_common_syn_eprop["optimizer"]["eta"] = eta
        nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

        params_gen_spk_in, params_gen_rate_target, params_gen_learning_window = get_params_task_input_output(
            self.n_iter_sim, loader
        )
        gen_spk_in.set(params_gen_spk_in)
        gen_rate_target.set(params_gen_rate_target)
        gen_learning_window.set(params_gen_learning_window)

        self.simulate("total_offset")
        self.simulate("extension_sim")

        if self.n_iter_sim > 0:
            self.evaluate()

        duration["sim"] = group_size * duration["sequence"] - duration["total_offset"] - duration["extension_sim"]

        self.simulate("sim")

        self.n_iter_sim += 1
        self.phase_label_previous = phase_label
        return


#%%

class eprop_model(Trainer):

    def __init__(self,batch_loader):
        Trainer.__init(self,batch_loader)
        self.in_shape = 0
        self.out_shape = 0
        self.rec_shape = 0

    def input_layer(self,n_in):
        self.in_shape = n_in
        self.in_layer = nest.Create('spike_generator',self.in_shape)

    def output_layer(self,n_out,params):
        self.out_shape = n_out
        self.out_layer = nest.Create('eprop_readout',self.out_shape,params)

    def recurrent_layer(self,n_rec,params,adaptive=False):
        self.rec_shape = n_rec
        if adaptive:
            model = 'eprop_iaf_adapt'
            assert "adapt_beta" in params, 'must define a threshold adaptation prefactor with key "adapt_beta"'
            assert "adapt_tau" in params, 'must define a threshold adaptation time constant (ms) with key "adapt_tau"'
        else:
            model = 'eprop_iaf'
        self.rec_layer = nest.Create(model,self.rec_shape,params)

    def compile(self,profiling='sparse',verbose=True):
        nest.ResetKernel()
        #nest.set(**params_setup)
        nest.Connect()
        nest.verbosity = nest.VerbosityLevel.FATAL

        if verbose:
            print("compiled")

    def classifier_signals(self,):
        return
#%%

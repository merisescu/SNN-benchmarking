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

    def evaluate(self):
        events_mm_out = self.mm_out.get("events")

        readout_signal = events_mm_out["readout_signal"]
        target_signal = events_mm_out["target_signal"]
        senders = events_mm_out["senders"]
        times = events_mm_out["times"]

        cond1 = times > (self.n_iter_sim - 1) * self.group_size * self.duration["sequence"] + self.duration["total_offset"]
        cond2 = times <= self.n_iter_sim * self.group_size * self.duration["sequence"] + self.duration["total_offset"]
        idc = cond1 & cond2

        readout_signal = np.array([readout_signal[idc][senders[idc] == i] for i in np.unique(senders)])
        target_signal = np.array([target_signal[idc][senders[idc] == i] for i in np.unique(senders)])

        readout_signal = readout_signal.reshape((self.out_shape, 1, self.group_size, self.steps["sequence"]))
        target_signal = target_signal.reshape((self.in_shape, 1, self.group_size, self.steps["sequence"]))

        readout_signal = readout_signal[:, :, :, -self.steps["learning_window"] :]
        target_signal = target_signal[:, :, :, -self.steps["learning_window"] :]

        loss = 0.5 * np.mean(np.sum((readout_signal - target_signal) ** 2, axis=3), axis=(0, 2))

        y_prediction = np.argmax(np.mean(readout_signal, axis=3), axis=0)
        y_target = np.argmax(np.mean(target_signal, axis=3), axis=0)
        accuracy = np.mean((y_target == y_prediction), axis=1)
        errors = 1.0 - accuracy

        self.results_dict["iteration"].append(self.n_iter_sim)
        self.results_dict["error"].extend(errors)
        self.results_dict["loss"].extend(loss)
        self.results_dict["label"].append(self.phase_label_previous)

        self.error = errors[0]

    def run_phase(self,phase_label,eta,loader):
        params_common_syn_eprop["optimizer"]["eta"] = eta
        nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

        params_gen_spk_in, params_gen_rate_target, params_gen_learning_window = self.classifier_signals(
            self.n_iter_sim, loader
        )
        self.gen_spk_in.set(params_gen_spk_in)
        self.gen_rate_target.set(params_gen_rate_target)
        self.gen_learning_window.set(params_gen_learning_window)

        self.simulate("total_offset")
        self.simulate("extension_sim")

        if self.n_iter_sim > 0 and self.profiling=="full":
            self.evaluate()

        self.duration["sim"] = self.group_size * self.duration["sequence"] - self.duration["total_offset"] - self.duration["extension_sim"]

        self.simulate("sim")

        self.n_iter_sim += 1
        self.phase_label_previous = phase_label
        return

    def run_training(self):
        self.run_phase("training",self.eta_train,self.batch_loader.training_loader)

    def run_validation(self):
        return

    def run_early_stopping(self):
        return

    def run_test(self):
        self.run_phase("test",0,self.batch_loader.test_loader)

    def fit(self,n_iter_train,group_size=100):
        self.group_size = group_size
        while self.k_iter < n_iter_train and not self.early_stop:
            self.run_validation()
            self.run_early_stopping()
            self.run_training()
            self.k_iter += 1
            print('epoch: ', self.k_iter)
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
        self.gen_spk_in = nest.Create("spike_generator", self.in_shape)
        self.nrns_in = nest.Create("parrot_neuron", self.in_shape)

    def output_layer(self,n_out,params):
        self.out_shape = n_out
        self.nrns_out = nest.Create('eprop_readout',self.out_shape,params)
        self.gen_rate_target = nest.Create("step_rate_generator", self.out_shape)
        self.gen_learning_window = nest.Create("step_rate_generator")

    def recurrent_layer(self,n_rec,params,adaptive=False):
        self.rec_shape = n_rec
        if adaptive:
            model = 'eprop_iaf_adapt'
            assert "adapt_beta" in params, 'must define a threshold adaptation prefactor with key "adapt_beta"'
            assert "adapt_tau" in params, 'must define a threshold adaptation time constant (ms) with key "adapt_tau"'
        else:
            model = 'eprop_iaf'

        scale_factor = 1.0 - params["kappa"]  # factor for rescaling due to removal of irregular spike arrival
        params["c_reg"] /= scale_factor**2 
        self.nrns_rec = nest.Create(model,self.rec_shape,params)
        self.eta_train = 5e-3 * scale_factor**2  # learning rate for training phase

    def set_timing(self,params):
        return

    def recorders(self):
        return

    def connect(self):
        return

    def compile(self,profiling='sparse',verbose=True):
        nest.ResetKernel()
        #nest.set(**params_setup)
        nest.Connect()
        nest.verbosity = nest.VerbosityLevel.FATAL

        if profiling=='sparse':
            params_mm_out = {
                "interval": self.duration["step"],
                "record_from": ["error_signal"],
                "start": self.duration["total_offset"],
                "label": "multimeter_out",
            }

        self.mm_out = nest.Create("multimeter", params_mm_out)
        
        if verbose:
            print("compiled")

    def classifier_signals(self):
        spike_times = [[] for _ in range(self.in_shape)]
        
        iteration_offset = self.n_iter_sim * self.group_size * self.duration["sequence"]
    
        params_gen_rate_target = [
            {
                "amplitude_times": np.arange(0.0, self.group_size * self.duration["sequence"], self.duration["sequence"])
                + iteration_offset
                + self.duration["total_offset"],
                "amplitude_values": np.zeros(self.group_size),
            }
            for _ in range(self.out_shape)
        ]
    
        for group_element in range(self.group_size):
            params_gen_rate_target[target_group[group_element]]["amplitude_values"][group_element] = 1.0
            for n, relative_times in enumerate(input_group[group_element]):
                if len(relative_times) > 0:
                    #print(n, relative_times)
                    relative_times = np.array(relative_times)
                    spike_times[n].extend(
                        iteration_offset + group_element * self.duration["sequence"] + relative_times + self.duration["offset_gen"]
                    )
                    # n is the pixel channel i think (flattened to 1d)
                    # relative spike times for each item is converted to absolute time in simulation
                    # iteration_offset - absolute amount of time since simulation start i.e. number of simulation trials (training, validation etc.) * size of each trial (group size * duration of each trial), then add group_element*duration['sequence'] to add how far along we are in this trial, then the offset_gen for the initial timer zeroing, ALL added to the relative time
    
        params_gen_learning_window = {
            "amplitude_times": np.hstack(
                [
                    np.array([0.0, self.duration["sequence"] - self.duration["learning_window"]])
                    + iteration_offset
                    + group_element * self.duration["sequence"]
                    + self.duration["total_offset"]
                    for group_element in range(self.group_size)
                ]
            ),
            "amplitude_values": np.tile([0.0, 1.0], self.group_size),
        }
    
        params_gen_spk_in = [{"spike_times": spk_times} for spk_times in spike_times]
    
        return params_gen_spk_in, params_gen_rate_target, params_gen_learning_window
#%%

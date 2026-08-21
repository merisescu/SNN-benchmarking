import numpy as np
import nest
import time
from sklearn.metrics import roc_auc_score
#%%

# we always want trainer to receive data as a list len(n_in) where each list item is an array that contains spike times for that neuron
# in order to allow for use with any dataset, we use a hook which should be defined outside the class to load each batch
class Trainer:

    def __init__(self):
        self.batch_loader = None
        self.results_dict = {
            "error": [],
            "auc": [],
            "spikes": [],
            "loss": [],
            "iteration": [],
            "label": [],
        }
        self.events = {}
        self.n_iter_sim = 0
        self.phase_label_previous = ""
        self.error = 0
        self.k_iter = 0
        self.early_stop = False
        self.profiling ='sparse'
        self.run_evaluation=False

    def evaluate(self):
        # if saving to disk, do the data analysis separately
        if not self.save_to_disk:
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
            target_signal = target_signal.reshape((self.out_shape, 1, self.group_size, self.steps["sequence"]))

            readout_signal = readout_signal[:, :, :, -self.steps["learning_window"] :]
            target_signal = target_signal[:, :, :, -self.steps["learning_window"] :]

            loss = 0.5 * np.mean(np.sum((readout_signal - target_signal) ** 2, axis=3), axis=(0, 2))

            # multiclass ROC area under curve score using one vs. rest

            y_prediction = np.mean(readout_signal, axis=3)
            y_target = np.mean(target_signal, axis=3)
            y_prediction = y_prediction.squeeze(1).T
            y_target = y_target.squeeze(1).T
            #print(y_prediction.shape, y_target.shape)
            auc = [roc_auc_score(y_target,y_prediction,average='micro',multi_class='ovr')]

            # binary accuracy

            y_prediction = np.argmax(np.mean(readout_signal, axis=3), axis=0)
            y_target = np.argmax(np.mean(target_signal, axis=3), axis=0)
            accuracy = np.mean((y_target == y_prediction), axis=1)
            errors = 1.0 - accuracy

            spikes = [self.sr_rec.n_events/self.group_size]
            # does not count spikes from the spike generator, this can be found either from self.sr_in.n_events
            # or take it from the data loader

            self.results_dict["iteration"].append(self.n_iter_sim)
            self.results_dict["error"].extend(errors)
            self.results_dict["auc"].extend(auc)
            self.results_dict["spikes"].extend(spikes)
            self.results_dict["loss"].extend(loss)
            self.results_dict["label"].append(self.phase_label_previous)

            self.error = errors[0]

    def run_phase(self,phase_label,eta,loader):
        
        self.params_common_syn_eprop["optimizer"]["eta"] = eta
        nest.SetDefaults("eprop_synapse", self.params_common_syn_eprop)

        params_gen_spk_in, params_gen_rate_target, params_gen_learning_window = self.classifier_signals(loader)
        self.gen_spk_in.set(params_gen_spk_in)
        self.gen_rate_target.set(params_gen_rate_target)
        self.gen_learning_window.set(params_gen_learning_window)

        self.simulate("total_offset")
        self.simulate("extension_sim")

        if self.n_iter_sim > 0 and self.run_evaluation:
            start = time.perf_counter()
            self.evaluate()
            if self.profiling=='full':
                self.get_events()
                print('got events')
            stop = time.perf_counter()
            print("evaluation time: ", stop-start)
        if not self.save_to_disk:
            nest.SetStatus(self.recorders, {'n_events': 0})
        
        self.duration["sim"] = self.group_size * self.duration["sequence"] - self.duration["total_offset"] - self.duration["extension_sim"]
        start = time.perf_counter()
        self.simulate("sim")
        stop = time.perf_counter()
        print("simulation time: ", stop-start)
        self.n_iter_sim += 1
        self.phase_label_previous = phase_label

        # this extra simulation shouldnt be needed
        #self.simulate("total_offset")
        #self.simulate("extension_sim")

        return

    def run_training(self):
        self.run_phase("training",self.eta_train,self.batch_loader.training_loader)

    def run_validation(self):
        self.run_phase("validation",0,self.batch_loader.validation_loader)

    def run_early_stopping(self):
        return

    def run_test(self):
        self.run_phase("test",0,self.batch_loader.test_loader)

    def fit(self,batch_loader,n_iter_train=200,group_size=100,n_validations=10):
        self.recorders = nest.GetNodes(properties={"element_type": "recorder"})
        self.group_size = group_size
        self.batch_loader = batch_loader
        self.batch_loader.group_size = group_size
        self.n_iter_train = n_iter_train
        while self.k_iter < self.n_iter_train:
            #self.run_validation()
            #self.run_early_stopping()
            self.run_training()
            self.k_iter += 1
            print('epoch: ', self.k_iter)
        self.run_evaluation=True
        for _ in range(n_validations):
            self.run_validation()

        self.simulate("total_offset")
        self.simulate("extension_sim")
        self.evaluate()
        self.get_events()

        self.duration["task"] = self.n_iter_sim * self.group_size * self.duration["sequence"] + self.duration["total_offset"]

        self.gen_spk_final_update.set({"spike_times": [self.duration["task"] + self.duration["extension_sim"] + 1]})

        self.simulate("final_update")
        return
        
    def simulate(self, k):
            nest.Simulate(self.duration[k])

    def get_events(self):
        # TODO: find memory leak
        def add_events(recorder_key, recorder):
            events = recorder.get('events')
            for k, v in events.items():
                self.events.setdefault(recorder_key, {}).setdefault(k, np.array([],dtype=v.dtype))
                self.events[recorder_key][k] = np.concatenate([self.events[recorder_key][k], np.asarray(v)])
        add_events("sr_in", self.sr_in)
        add_events("sr_rec", self.sr_rec)
        add_events("mm_rec", self.mm_rec)
        add_events("mm_out", self.mm_out)
        add_events("wr", self.wr)

    def get_results(self):
        for k, v in self.results_dict.items():
            self.results_dict[k] = np.array(v)
        return self.results_dict, self.events


#%%

class eprop_model(Trainer):

    def __init__(self,num_threads,num_virtual_procs):
        Trainer.__init__(self)
        self.verbose = True
        self.in_shape = 0
        self.out_shape = 0
        self.rec_shape = 0
        self.params_setup = { "total_num_virtual_procs": num_virtual_procs}
        nest.ResetKernel()
        nest.verbosity = nest.VerbosityLevel.FATAL
        nest.local_num_threads = num_threads

    
    def timing(self,resolution,delay,params):
        self.params_setup.update({"resolution": resolution})
        nest.set(**self.params_setup)
        params.update({
            "offset_gen": 1,
            "delay_in_rec": delay,
            "extension_sim": 1,
            "final_update": 3
        })
        params['delays'] = params['delay_in_rec'] # keep both keys just in case

        params['total_offset'] = params['offset_gen'] + params['delays']

        self.steps = params

        self.duration = {'step': resolution}

        self.duration.update({key: value * self.duration['step'] for key, value in params.items()})

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
        self.adaptive = adaptive
        self.voltage_scale = (params['V_th']-params['E_L'])/0.6
        if adaptive:
            model = 'eprop_iaf_adapt'
        else:
            model = 'eprop_iaf'

        self.scale_factor = 1.0 - params["kappa"]  # factor for rescaling due to removal of irregular spike arrival

        params["flush_event_send_interval"] = self.duration["sequence"]
        self.nrns_rec = nest.Create(model,self.rec_shape,params)

    def recorders(self,neurons_sampled=1,synapses_sampled=5,profiling='sparse',save_to_disk=True):
        self.n_record = neurons_sampled
        self.n_record_w = synapses_sampled
        self.profiling = profiling
        self.save_to_disk = save_to_disk
        if save_to_disk:
            nest.SetKernelStatus({
                "data_path": "./sion/",
                "recording_backends": {
                    "sionlib": {
                        "filename": "run01.sion",
                        "buffer_size": 4096,
                        "sion_n_files": 4,
                        "sion_collective": True
                    }
                }
            })
        else:
            self.backend = 'memory'
        start = self.duration['total_offset']
        params_mm_out = {
            "interval": self.duration["step"],
            "record_to": self.backend,
            "record_from": ["V_m", "readout_signal", "target_signal", "error_signal"],
            "start": start,
            "label": "multimeter_out",
        }
        
        params_mm_rec = {
            "interval": self.duration["step"],  # interval between two recorded time points
            "record_to": self.backend,
            "record_from": ["V_m", "surrogate_gradient", "learning_signal"],  # dynamic variables to record
            "start": start,  # start time of recording
            "label": "multimeter_rec",
        }

        if self.adaptive:
            params_mm_rec['record_from'].extend(["V_th_adapt"])

        params_wr = {
            "senders": self.nrns_in[:self.n_record_w] + self.nrns_rec[:self.n_record_w],  # limit senders to subsample weights to record
            "targets": self.nrns_rec[:self.n_record_w] + self.nrns_out,  # limit targets to subsample weights to record from
            "record_to": self.backend,
            "start": start,
            "label": "weight_recorder",
        }

        params_sr_in = {
            "record_to": self.backend,
            "start": start,
            "label": "spike_recorder_in",
        }

        params_sr_rec = {
            "record_to": self.backend,
            "start": start,
            "label": "spike_recorder_rec",
        }

        if profiling=='sparse':
            #params_sr_in['record_to'] = ""
            #params_sr_rec['record_to'] = ""
            params_mm_out['record_from'] = ["readout_signal", "target_signal"]
            self.sr_in = nest.Create("spike_recorder", params_sr_in)
            self.mm_rec = nest.Create("multimeter", params_mm_rec)
        elif profiling=='full':
            self.run_evaluation=True
            self.sr_in = nest.Create("spike_recorder", params_sr_in)
            # leaving here for now bc only benchmarking off spikes
            self.mm_rec = nest.Create("multimeter", params_mm_rec)


        self.wr = nest.Create("weight_recorder", params_wr)

        self.mm_out = nest.Create("multimeter", params_mm_out)
        self.sr_rec = nest.Create("spike_recorder", params_sr_rec)
        self.nrns_rec_record = self.nrns_rec[:self.n_record]

        return

    def connect(self,sparsity=[0.9,0.99,0],eta_train=5e-3,eta_test=0,optimizer='gradient_descent'):
        params_conn_all_to_all = {"rule": "all_to_all", "allow_autapses": False}
        params_conn_one_to_one = {"rule": "one_to_one"}

        def calculate_glorot_dist(fan_in, fan_out):
            glorot_scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
            glorot_limit = np.sqrt(3.0 * glorot_scale)
            glorot_distribution = np.random.uniform(low=-glorot_limit, high=glorot_limit, size=(fan_in, fan_out))
            return glorot_distribution


        def create_mask(weights, sparsity_level):
            return np.random.choice([0, 1], weights.shape, p=[sparsity_level, 1 - sparsity_level])


        def get_weight_recorder_senders_targets(weights, sender_pop, target_pop):
            target_idc, sender_idc = np.where(weights)
            senders = sender_pop[np.unique(sender_idc[:self.n_record_w])]
            targets = target_pop[np.unique(target_idc[:self.n_record_w])]
            return senders, targets

        weights_in_rec = np.array(np.random.randn(self.in_shape, self.rec_shape).T / np.sqrt(self.in_shape)) * self.voltage_scale
        weights_rec_rec = np.array(np.random.randn(self.rec_shape, self.rec_shape).T / np.sqrt(self.rec_shape)) * self.voltage_scale
        np.fill_diagonal(weights_rec_rec, 0.0)  # since no autapses set corresponding weights to zero
        weights_rec_out = np.array(calculate_glorot_dist(self.rec_shape, self.out_shape).T) * self.scale_factor
        weights_out_rec = np.array(np.random.randn(self.rec_shape, self.out_shape)) * self.voltage_scale / self.scale_factor

        sparsity_level_in_rec = sparsity[0]
        sparsity_level_rec_rec = sparsity[1]
        sparsity_level_rec_out = sparsity[2]

        weights_in_rec *= create_mask(weights_in_rec, sparsity_level_in_rec)
        weights_rec_rec *= create_mask(weights_rec_rec, sparsity_level_rec_rec)
        weights_rec_out *= create_mask(weights_rec_out, sparsity_level_rec_out)

        senders_in_rec, targets_in_rec = get_weight_recorder_senders_targets(weights_in_rec, self.nrns_in, self.nrns_rec)
        senders_rec_rec, targets_rec_rec = get_weight_recorder_senders_targets(weights_rec_rec, self.nrns_rec, self.nrns_rec)
        senders_rec_out, targets_rec_out = get_weight_recorder_senders_targets(weights_rec_out, self.nrns_rec, self.nrns_out)
        self.wr.set(
            senders=np.unique(np.concatenate([senders_in_rec, senders_rec_rec, senders_rec_out])),
            targets=np.unique(np.concatenate([targets_in_rec, targets_rec_rec, targets_rec_out])),
        )

        self.params_common_syn_eprop = {
            "optimizer": {
                "type": optimizer,  # algorithm to optimize the weights
                "batch_size": 1, # probably leave this at 1, this just witholds updates, doesnt actually parallelise anything
                "optimize_each_step": False,  # call optimizer every time step (True) or once per spike (False); both
                # yield same results for gradient descent, False offers speed-up
                "Wmin": -10000.0,  # pA, minimal limit of the synaptic weights
                "Wmax": 10000.0,  # pA, maximal limit of the synaptic weights
            },
        }

        self.params_common_syn_eprop["weight_recorder"] = self.wr

        self.eta_test = eta_test  # learning rate for test phase
        self.eta_train = eta_train * self.scale_factor**2  # learning rate for training phase

        params_syn_base = {
            "synapse_model": "eprop_synapse",
            "delay": self.duration["step"],  # ms, dendritic delay
        }

        params_syn_in = params_syn_base.copy()
        params_syn_rec = params_syn_base.copy()
        params_syn_out = params_syn_base.copy()

        params_syn_feedback = {
            "synapse_model": "eprop_learning_signal_connection",
            "delay": self.duration["step"],
            "weight": weights_out_rec,
        }

        params_syn_learning_window = {
            "synapse_model": "rate_connection_delayed",
            "delay": self.duration["step"],
            "receptor_type": 1,  # receptor type over which readout neuron receives learning window signal
        }

        params_syn_rate_target = {
            "synapse_model": "rate_connection_delayed",
            "delay": self.duration["step"],
            "receptor_type": 2,  # receptor type over which readout neuron receives target signal
        }

        params_syn_static = {
            "synapse_model": "static_synapse",
            "delay": self.duration["step"],
        }

        nest.SetDefaults("eprop_synapse", self.params_common_syn_eprop)
        nest.Connect(self.gen_spk_in, self.nrns_in, params_conn_one_to_one, params_syn_static) # connection 1

        def sparsely_connect(weights, params_syn, nrns_pre, nrns_post):
            targets, sources = np.where(weights)
            params_syn["weight"] = weights[targets, sources].flatten()
            params_syn["delay"] = [params_syn["delay"] for _ in params_syn["weight"]]
            nrns_pre_arr = np.array(nrns_pre.tolist())
            nrns_post_arr = np.array(nrns_post.tolist())
            nest.Connect(nrns_pre_arr[sources], nrns_post_arr[targets], params_conn_one_to_one, params_syn)

        sparsely_connect(weights_in_rec, params_syn_in, self.nrns_in, self.nrns_rec)  # connection 2
        sparsely_connect(weights_rec_rec, params_syn_rec, self.nrns_rec, self.nrns_rec)  # connection 3
        sparsely_connect(weights_rec_out, params_syn_out, self.nrns_rec, self.nrns_out)  # connection 4

        nest.Connect(self.nrns_out, self.nrns_rec, params_conn_all_to_all, params_syn_feedback)  # connection 5
        nest.Connect(self.gen_rate_target, self.nrns_out, params_conn_one_to_one, params_syn_rate_target)  # connection 6
        nest.Connect(self.gen_learning_window, self.nrns_out, params_conn_all_to_all, params_syn_learning_window)  # connection 7

        if self.profiling == 'full' or self.profiling == 'sparse':
            nest.Connect(self.nrns_in, self.sr_in, params_conn_all_to_all, params_syn_static)
            nest.Connect(self.nrns_rec, self.sr_rec, params_conn_all_to_all, params_syn_static)
            nest.Connect(self.mm_rec, self.nrns_rec_record, params_conn_all_to_all, params_syn_static)
            nest.Connect(self.mm_out, self.nrns_out, params_conn_all_to_all, params_syn_static)
        else:
            nest.Connect(self.nrns_rec, self.sr_rec, params_conn_all_to_all, params_syn_static)
            nest.Connect(self.mm_out, self.nrns_out, params_conn_all_to_all, params_syn_static)

        self.gen_spk_final_update = nest.Create("spike_generator", 1)
        nest.Connect(self.gen_spk_final_update, self.nrns_in + self.nrns_rec, "all_to_all", {"weight": 1000.0})

    def compile(self,verbose=True):
        self.verbose = verbose
        if verbose:
            weights_pre_train = {
                "in_rec": self.get_weights(self.nrns_in, self.nrns_rec),
                "rec_rec": self.get_weights(self.nrns_rec, self.nrns_rec),
                "rec_out": self.get_weights(self.nrns_rec, self.nrns_out),
            }
            return weights_pre_train

    def get_weights(self,pop_pre, pop_post):
        conns = nest.GetConnections(pop_pre, pop_post).get(["source", "target", "weight"])
        conns["senders"] = np.array(conns["source"]) - np.min(conns["source"])
        conns["targets"] = np.array(conns["target"]) - np.min(conns["target"])

        conns["weight_matrix"] = np.zeros((len(pop_post), len(pop_pre)))
        conns["weight_matrix"][conns["targets"], conns["senders"]] = conns["weight"]
        return conns

    def classifier_signals(self,loader):
        start = time.perf_counter()
        input_group, target_group = loader()
        stop = time.perf_counter()
        print("data loading time: ", stop-start)
        # TODO: remove hardcoding of t_max = 336040
        start = time.perf_counter()
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
                    relative_times = np.array(relative_times)
                    relative_times = np.around(relative_times * self.duration['sequence'] / 336040)
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

        stop = time.perf_counter()
        print("data to signals time: ", stop-start)
    
        return params_gen_spk_in, params_gen_rate_target, params_gen_learning_window
#%%

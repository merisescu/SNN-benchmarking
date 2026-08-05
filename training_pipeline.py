

#%%

class TrainingPipeline:
    def __init__(self,dataset,metrics,recording=='sparse'):
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

        self.current_index = 0

    def se
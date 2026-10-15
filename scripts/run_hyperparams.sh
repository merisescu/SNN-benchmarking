#!/bin/bash

.venv/bin/python -m scripts.init_studies

parallel -u -j $1 -D 2 .venv/bin/python -m ::: \
  $(yes scripts.LIF_MNIST_hyperparameter_search | head -$2) \
  $(yes scripts.ALIF_MNIST_hyperparameter_search | head -$3) \
  $(yes scripts.LIF_MG_hyperparameter_search | head -$4) \
  $(yes scripts.ALIF_MG_hyperparameter_search | head -$5) \
  $(yes scripts.Izh_MG_hyperparameter_search | head -$6) \
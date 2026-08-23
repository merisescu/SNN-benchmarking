#!/bin/bash

.venv/bin/python -m scripts.init_studies

parallel -u -j $1 -D 2 .venv/bin/python -m ::: \
  $(yes scripts.LIF_hyperparameter_search | head -$2) \
  $(yes scripts.ALIF_hyperparameter_search | head -$3)

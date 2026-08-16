#!/bin/bash

python -m scripts.init_studies

parallel --ungroup -j $1 -D 5 scripts.LIF_hyperparameter_search.py ::: 1

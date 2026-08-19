#!/bin/bash

.venv/bin/python -m scripts.init_studies

parallel --ungroup -j $1 -D 5 .venv/bin/python -m scripts.LIF_hyperparameter_search.py ::: {1..$1}

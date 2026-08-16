#!/bin/bash

python -m scipts.init_studies

parallel -j $1 scripts.LIF_hyperparameter_search.py ::: {1..$2}

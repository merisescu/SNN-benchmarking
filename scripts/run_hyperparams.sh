#!/bin/bash

.venv/bin/python -m scripts.init_studies

JOBS=$1
JOBS_PER= $1 / 2

{
    seq JOBS_PER | sed 's|^|scripts.LIF_hyperparameters_search |'
    seq JOBS_PER | sed 's|^|scripts.ALIF_hyperparameters_search |'
} | parallel -j JOBS --colsep ' './venv/bin/python -m {1}

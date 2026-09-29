#!/bin/bash

# run job
../../build/NQS \
    --group_name c2v \
    --n_sample 20000 --sample_thresh 1e-4 \
    --max_iter 1000 \
    --alpha 6.0 \
    --n_block 2 \
    --max_lr 0.2 --min_lr 1e-5 --diag_eps 1e-4\
    --n_search 5 \
    --subspace_iter 4 \
    --inv_temp 2.0 

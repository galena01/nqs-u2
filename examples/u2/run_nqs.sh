#!/bin/bash

# run job
../../build/NQS \
  --seed 11 \
  --init_sample mcmc \
  --group_name d2h \
  --mol_itgs_file u2.FciDmp \
  --load_checkpoint pretrain_B1g_25_0.npz \
  --n_sample 200000 \
  --sample_thresh 1e-4 \
  --max_iter 243 \
  --alpha 10.0 \
  --n_block 5 \
  --max_lr 0.2 \
  --min_lr 0.00001 \
  --n_search 10 \
  --diag_eps 1e-4 \
  --subspace_iter 4 \
  --inv_temp 0.0 \
  --mcmc_iter 0 \
  --save_checkpoint u2_B1g_
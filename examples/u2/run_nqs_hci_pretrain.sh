#!/bin/bash

# run job
../../build/NQS \
  --seed 11 \
  --init_sample file \
  --init_subspace_file hci_B1g_init.txt \
  --guide_ci_file hci_B1g_init.txt \
  --guide_weight 0.1 \
  --guide_stop_macro 25 \
  --group_name d2h \
  --mol_itgs_file u2.FciDmp  \
  --n_sample 3000 \
  --sample_thresh 1e-3 \
  --max_iter 200 \
  --alpha 10.0 \
  --n_block 5 \
  --max_lr 0.2 \
  --min_lr 0.00001 \
  --n_search 10 \
  --diag_eps 1e-4 \
  --subspace_iter 4 \
  --inv_temp 0.0 \
  --mcmc_iter 0 \
  --save_checkpoint pretrain_B1g_
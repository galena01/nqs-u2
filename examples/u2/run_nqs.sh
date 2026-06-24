#!/bin/bash

# run job
../../build/NQS   --seed 42    --group_name d2h   --mol_itgs_file u2.FciDmp   --n_sample 100000   --sample_thresh 1e-4   --max_iter 1000   --alpha 10.0   --n_block 5   --max_lr 0.2   --min_lr 0.00001   --n_search 10   --eps 1e-4   --subspace_iter 4   --inv_temp 0.0   --mcmc_iter 0   --save_checkpoint "ckpt_"   --load_checkpoint "pretrain_75_0.npz"   

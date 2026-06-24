#pragma once

#include "RBM.h"
#include "utils/Slater.h"

vector<SlaterInt_t> mcmc_sampling(int n_chain, int n_step, int sample_interval, int burn_in, const NN_Params &p);

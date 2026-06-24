#pragma once

#include <Eigen/Dense>
#include "RBM.h"
#include "utils/Slater.h"

Eigen::VectorXcd sr(NN_Params &params_NN, vector<SlaterInt_t> xint,
                    Eigen::VectorXd rho, Eigen::VectorXcd elocs, int nCasOrb,
                    double eps, Eigen::dcomplex eng, int n_block);

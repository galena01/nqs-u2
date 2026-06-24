#pragma once

#include "utils/Slater.h"
#include "RBM.h"
#include "Eigen/Sparse"

using WfnMap_t = unordered_map<SlaterInt_t, Eigen::dcomplex, SlaterIntHash>;

Eigen::dcomplex eloc_subspace(SlaterInt_t state, NN_Params &pnet, const WfnMap_t &logwfn_hash);

Eigen::VectorXcd omp_eloc_subspace(const vector<SlaterInt_t> &states, NN_Params &pnet, const WfnMap_t &logwfn_hash);

pair<Eigen::VectorXcd, WfnMap_t> omp_eloc(const vector<SlaterInt_t> &states, NN_Params &pnet, int target_n_state, double sc_thresh, bool sc_flag);

Eigen::dcomplex eloc(SlaterInt_t state, NN_Params &pnet, WfnMap_t &sc_selector);

Eigen::SparseMatrix<double> get_H_subspace(const vector<SlaterInt_t> &subspace);

pair<Eigen::dcomplex, Eigen::VectorXcd> get_energy_subspace_new(const Eigen::SparseMatrix<double> &H_subspace, const Eigen::VectorXcd &psi, const Eigen::VectorXd &rho);

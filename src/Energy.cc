#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#include "Energy.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <Eigen/Eigenvalues>
#include "utils/utils.h"

#include "utils/Slater.h"
#include "RBM.h"
#include "MolLoader.h"
#include "Hamiltonian.h"


struct KVpod {
    uint8_t alpha[MAX_SLATER_SIZE];
    uint8_t beta [MAX_SLATER_SIZE];
    double  re;
    double  im;
};



#define HAM_THRESH              1e-10

Eigen::dcomplex eloc_subspace(SlaterInt_t state, NN_Params &pnet, const WfnMap_t &logwfn_hash)
{
    const Molecule& mol = Molecule::get_instance();
    vector<SlaterInt_t> sd = sd_space(state, mol.irrep_ids, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.target_irrep_ids);
    Eigen::dcomplex logwfn_0 = logwfn_hash.at(sd[0]);
    Eigen::dcomplex logwfn_i;
    Eigen::dcomplex rtn=0;
    for(int i=0;i<sd.size();i++){
        double Hi0 = Mat_element_from_state_int(sd[i],sd[0]);
        if(logwfn_hash.count(sd[i])!=0){
            Eigen::dcomplex ratio = exp(logwfn_hash.at(sd[i])-logwfn_0);
            rtn += ratio * Hi0;
        }
    }

    return rtn;
}

Eigen::VectorXcd omp_eloc_subspace(const vector<SlaterInt_t> &states, NN_Params &pnet, const WfnMap_t &logwfn_hash)
{
    int n_state = states.size();
    Eigen::VectorXcd elocs(n_state);

    #pragma omp parallel for
    for(int i=0;i<n_state;i++){
        elocs[i] = eloc_subspace(states[i], pnet, logwfn_hash);
    }

    return elocs;
}


Eigen::dcomplex eloc(SlaterInt_t state, NN_Params &pnet, WfnMap_t &sc_selector, bool sc_flag, double sc_thresh)
{
    const Molecule& mol = Molecule::get_instance();
    vector<SlaterInt_t> sd = sd_space(state, mol.irrep_ids, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.target_irrep_ids);
    Eigen::VectorXcd logpsis = NN_forward_batch_eloc(pnet, sd, mol.nCasOrb);
    Eigen::dcomplex rtn=0;
    for(int i=0;i<sd.size();i++){
        double Hi0 = Mat_element_from_state_int(sd[i],sd[0]);
        Eigen::dcomplex ratio = exp(logpsis[i]-logpsis[0]);
        rtn += ratio * Hi0;

        if(sc_flag && logpsis[i].real() > sc_thresh){
            sc_selector[sd[i]] = logpsis[i];
        }
    }

    return rtn;
}

pair<Eigen::VectorXcd, WfnMap_t> omp_eloc(const vector<SlaterInt_t> &states, NN_Params &pnet, int target_n_state, double sc_thresh, bool sc_flag)
{

    int n_state = states.size();
    Eigen::VectorXcd elocs(n_state);
    elocs.setZero();
    int N_THREAD = Eigen::nbThreads();

    vector<WfnMap_t> sc_selector_parallel;
    WfnMap_t sc_selector_full;

    sc_selector_parallel.resize(N_THREAD);

    int NUM_PER_LOOP = 2048;
    int n_loop = (n_state-1)/NUM_PER_LOOP + 1;

    cout << "    " << strtime() << "Computing elocs ..."  << endl;

    for(int i=0;i<n_loop;i++){
        int start = i * NUM_PER_LOOP;
        int end = min((i+1)* NUM_PER_LOOP, n_state);

        #pragma omp parallel for
        for(int i=start;i<end;i++){
            assert(N_THREAD==omp_get_num_threads());
            int tid = omp_get_thread_num();
            elocs[i] = eloc(states[i], pnet, sc_selector_parallel[tid], sc_flag, sc_thresh);
        }

        for(int tid=0;tid<N_THREAD;tid++){
            for(auto kv:sc_selector_parallel[tid]){
                sc_selector_full[kv.first] = kv.second;
            }
            sc_selector_parallel[tid].clear();
        }

    }



    if(!sc_flag){
        return make_pair(elocs, WfnMap_t());
    }

    cout << "    " << strtime() << "Selecting subspacce ..." << endl;



    vector<pair<SlaterInt_t, Eigen::dcomplex>> sc_vec;
    for (const auto& kv : sc_selector_full) {
        sc_vec.push_back(make_pair(kv.first, kv.second));
    }

    WfnMap_t selected_subspace_logwfn;
    if (target_n_state < 0){
        for (const auto& kv : sc_selector_full) {
            selected_subspace_logwfn.insert(make_pair(kv.first, kv.second));
        }
    }else{
        target_n_state = min(target_n_state, (int)sc_vec.size());

        std::nth_element(
            sc_vec.begin(), sc_vec.begin() + target_n_state - 1, sc_vec.end(),
            [](const auto& a, const auto& b) { return a.second.real() > b.second.real(); }
        );

        for(int i=0; i<target_n_state; i++){
            selected_subspace_logwfn.insert(make_pair(sc_vec[i].first, sc_vec[i].second));
        }
    }

    return make_pair(elocs, selected_subspace_logwfn);
}


pair<Eigen::dcomplex, Eigen::VectorXcd> get_energy_subspace_new(const Eigen::SparseMatrix<double> &H_subspace, const Eigen::VectorXcd &psi, const Eigen::VectorXd &rho)
{
    Eigen::VectorXcd H_psi = H_subspace * psi;

    Eigen::VectorXcd elocs = Eigen::VectorXcd::Zero(psi.size());

    elocs = H_psi.cwiseQuotient(psi);

    Eigen::dcomplex energy = rho.dot(elocs);

    return make_pair(energy, elocs);
}

Eigen::SparseMatrix<double> get_H_subspace(const vector<SlaterInt_t> &subspace)
{
    Eigen::SparseMatrix<double> subspace_H;
    std::vector<Eigen::Triplet<double>> tripletList;

    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<double>> local_triplets;

        #pragma omp for nowait
        for(size_t i=0; i<subspace.size(); i++){
            for(size_t j=0; j<subspace.size(); j++){
                double val = Mat_element_from_state_int(subspace[i], subspace[j]);
                if(std::fabs(val) > 1e-12){
                    local_triplets.push_back(Eigen::Triplet<double>(i, j, val));
                }
            }
        }

        #pragma omp critical
        tripletList.insert(tripletList.end(), local_triplets.begin(), local_triplets.end());
    }

    subspace_H.resize(subspace.size(), subspace.size());
    subspace_H.setFromTriplets(tripletList.begin(), tripletList.end());
    subspace_H.makeCompressed();
    return subspace_H;
}

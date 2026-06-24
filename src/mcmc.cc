#include <unordered_set>

#include "utils/Slater.h"
#include "utils/utils.h"
#include "MolLoader.h"
#include "RBM.h"

using namespace std;

#define MIN_LOGPSI -1e8

SlaterInt_t exci1e(SlaterInt_t state, int nCasOrb)
{
    SlaterInt_t new_slater = state;

    auto &engine = oRNG.engine();
    uniform_int_distribution<int> dist(0, 1);
    int choice = dist(engine);
    uint8_t* orbs = (choice == 0) ? new_slater.alpha : new_slater.beta;

    vector<int> occupied, virtuals;
    for(int i=0; i<nCasOrb; i++){
        if(get_bit(orbs, i)){
            occupied.push_back(i);
        }else{
            virtuals.push_back(i);
        }
    }
    if(occupied.empty() || virtuals.empty()){
        throw runtime_error("No occupied or virtual orbitals available for excitation.");
    }

    int from, to;
    uniform_int_distribution<int> occ_dist(0, occupied.size() - 1);
    uniform_int_distribution<int> virt_dist(0, virtuals.size() - 1);
    from = occupied[occ_dist(engine)];
    to = virtuals[virt_dist(engine)];

    set_bit(orbs, from, 0);
    set_bit(orbs, to, 1);

    return new_slater;

}

SlaterInt_t kick(const SlaterInt_t& slater, const NN_Params &p)
{
    Molecule &mol = Molecule::get_instance();
    SlaterInt_t new_slater;
    new_slater = exci1e(slater, mol.nCasOrb);
    new_slater = exci1e(new_slater, mol.nCasOrb);

    auto &engine = oRNG.engine();
    Eigen::VectorXcd logwfn = NN_forward_loop(p, {slater, new_slater}, mol.nCasOrb);


    if(!spa_symm_allowed(new_slater, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.irrep_ids, mol.target_irrep_ids)){
        logwfn[1] = MIN_LOGPSI;
    }

    if(!spa_symm_allowed(slater, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.irrep_ids, mol.target_irrep_ids)){
        logwfn[0] = MIN_LOGPSI;
    }

    double log_acc_ratio = 2.0 * (logwfn[1].real() - logwfn[0].real());
    uniform_real_distribution<double> u_dist(0.0, 1.0);
    if(log(u_dist(engine)) < log_acc_ratio){
        return new_slater;
    }else{
        return slater;
    }

}

vector<SlaterInt_t> mcmc_sampling(int n_chain, int n_step, int sample_interval, int burn_in, const NN_Params &p)
{
    Molecule &mol = Molecule::get_instance();

    SlaterInt_t hf = hf_state(mol.nCasAlpha, mol.nCasBeta);
    vector<unordered_set<SlaterInt_t, SlaterIntHash>> local_sets(n_chain);

    #pragma omp parallel for schedule(static)
    for(int i=0; i<n_chain; i++){
        SlaterInt_t slater = hf;
        for(int j=0; j<n_step; j++){
            slater = kick(slater, p);
            if(j % sample_interval == 0){
                if(spa_symm_allowed(slater, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.irrep_ids, mol.target_irrep_ids)){
                    local_sets[i].insert(slater);
                }
            }
        }
    }

    unordered_set<SlaterInt_t, SlaterIntHash> all_samples;
    for(auto& s : local_sets){
        all_samples.insert(s.begin(), s.end());
    }

    vector<SlaterInt_t> samples(all_samples.begin(), all_samples.end());

    return samples;

}

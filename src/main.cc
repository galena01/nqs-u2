
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <unordered_map>

#include <pybind11/embed.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "utils/args.h"
#include "utils/Slater.h"
#include "utils/utils.h"
#include "utils/pyutils.h"

#include "MolLoader.h"
#include "Hamiltonian.h"
#include "RBM.h"
#include "Energy.h"
#include "SROpt.h"
#include "mcmc.h"


namespace py = pybind11;
using namespace std;

using GuideMap_t = unordered_map<SlaterInt_t, double, SlaterIntHash>;

static GuideMap_t load_guide_ci(const string &filename, int nCasOrb)
{
    ifstream fin(filename);
    if(!fin) throw runtime_error("Cannot open HCI guide file: " + filename);

    GuideMap_t guide;
    string line;
    double norm2 = 0.0;
    while(getline(fin, line)) {
        if(line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        SlaterInt_t state{};
        for(int i = 0; i < nCasOrb; ++i) {
            string token;
            if(!(iss >> token)) throw runtime_error("Malformed HCI guide line: " + line);
            if(token == "2") {
                set_bit(state.alpha, i, 1);
                set_bit(state.beta, i, 1);
            } else if(token == "u") {
                set_bit(state.alpha, i, 1);
            } else if(token == "d") {
                set_bit(state.beta, i, 1);
            } else if(token != "0") {
                throw runtime_error("Invalid occupation token in HCI guide: " + token);
            }
        }
        double coefficient;
        if(!(iss >> coefficient)) throw runtime_error("Missing coefficient in HCI guide line: " + line);
        guide[state] = coefficient;
        norm2 += coefficient * coefficient;
    }
    if(guide.empty() || !(norm2 > 0.0)) throw runtime_error("Empty HCI guide: " + filename);
    const double inv_norm = 1.0 / sqrt(norm2);
    for(auto &entry : guide) entry.second *= inv_norm;
    return guide;
}

static pair<double, Eigen::VectorXcd> guide_projector_local_energy(
    const vector<SlaterInt_t> &subspace, const Eigen::VectorXcd &wfn,
    const GuideMap_t &guide)
{
    Eigen::VectorXd target = Eigen::VectorXd::Zero(subspace.size());
    for(size_t i = 0; i < subspace.size(); ++i) {
        auto it = guide.find(subspace[i]);
        if(it != guide.end()) target[i] = it->second;
    }

    const Eigen::dcomplex overlap = target.cast<Eigen::dcomplex>().dot(wfn);
    const double wfn_norm2 = wfn.squaredNorm();

    const double fidelity = norm(overlap) / wfn_norm2;

    Eigen::VectorXcd local = Eigen::VectorXcd::Ones(subspace.size());
    for(int i = 0; i < local.size(); ++i) {
        if(target[i] != 0.0) local[i] -= target[i] * overlap / wfn[i];
    }
    return {fidelity, local};
}

int main(int argc,char* argv[])
{
    Args_t args = parse_args(argc, argv);

    py::scoped_interpreter guard{};
    cout<<setprecision(10);

    oRNG.init(args.n_thread, args.seed);

    Molecule& mol = Molecule::get_instance();
    
    if(mol.load_mol_pyscf(args.mol_info_file)==0){
        cout << "Loaded PySCF molecule information from " << args.mol_info_file << endl;
        mol.load_fcidump_pyscf(args.mol_itgs_file);
    }else{
        cout << "No PySCF molecule information found, assuming molcas input" << endl;
        assert(args.group_name != "none");
        for(auto & c : args.group_name) {c = tolower(c);}
        cout << "Using group name: " << args.group_name << endl;
        mol.load_fcidump_molcas(args.mol_itgs_file, args.group_name);
    }
    cout << mol << endl;
    
    SlaterInt_t hf = hf_state(mol.nCasAlpha, mol.nCasBeta);
    cout << "HF state: ";
    print_slater(hf, mol.nCasOrb);
    cout << endl;

    NN_Params params_NN(mol.nCasOrb*2, args.alpha, mol.mo_energy, args.inv_temp);

    GuideMap_t guide_ci;
    if(args.guide_weight > 0.0) {
        guide_ci = load_guide_ci(args.guide_ci_file, mol.nCasOrb);
        cout << "Loaded " << guide_ci.size() << " normalized HCI guide coefficients." << endl;
    }

    vector<SlaterInt_t> subspace;
    vector<SlaterInt_t> subspace_file;
    Eigen::SparseMatrix<double> H_subspace;

    if(args.init_sample == "random"){
        cout << "Generating random subspace..." << endl;
        unordered_set<SlaterInt_t, SlaterIntHash> subspace_set;
        for(int i=0; i<args.n_sample; i++){
            SlaterInt_t s = hf;
            shuffle_bits(s.alpha, mol.nCasOrb, oRNG.engine());
            if(spa_symm_allowed(s, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.irrep_ids, mol.target_irrep_ids)){
                subspace_set.insert(s);
            }
        }
        for(auto s : subspace_set){
            subspace.push_back(s);
        }

    }else if(args.init_sample == "cisd"){
        vector<SlaterInt_t> sd_subspace = sd_space(hf, mol.irrep_ids, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.target_irrep_ids);

        cout << "SD space size: " << sd_subspace.size() << endl;
        H_subspace = get_H_subspace(sd_subspace);

        Spectra::SparseGenMatProd<double> op(H_subspace);
        Spectra::SymEigsSolver<Spectra::SparseGenMatProd<double>> eigs(op, 1, min(64, (int)sd_subspace.size()));
        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 10000, 1e-10);
        Eigen::VectorXd eigenvalues;
        if(eigs.info() == Spectra::CompInfo::Successful){
            eigenvalues = eigs.eigenvalues();
            std::cout << "CISD Ground state energy: " << eigenvalues(0) + mol.e_core << std::endl;
        }else{
            std::cerr << "CISD computation failed!" << std::endl;
        }

        for(auto &s: sd_subspace){
            if(spa_symm_allowed(s, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.irrep_ids, mol.target_irrep_ids)){
                subspace.push_back(s);
            }
        }
    }else if(args.init_sample == "mcmc"){
        int n_chain = args.n_thread * 10;
        int n_step = args.n_sample / n_chain + 300;
        subspace = mcmc_sampling(n_chain, n_step, 2*mol.nCasOrb, 10*mol.nCasOrb, params_NN);
    }else if(args.init_sample == "file"){
            if(args.init_subspace_file.empty()){
                std::cerr << "Error: init_sample = 'file' but --init_subspace_file is not provided." << std::endl; std::exit(1);
            }
            std::cout << "Loading initial subspace from file: " 
                      << args.init_subspace_file << std::endl;
            subspace = load_subspace_from_file(
                args.init_subspace_file,
                mol.nCasOrb
            );
            for(int i=0; i<subspace.size(); i++){
                auto s = subspace[i];
                if(spa_symm_allowed(s, mol.nCasOrb, mol.nCasAlpha, mol.nCasBeta, mol.groupname, mol.irrep_ids, mol.target_irrep_ids)){
                    subspace_file.push_back(s);
                }
            }

            std::cout << subspace_file.size() << " of " << subspace.size() << " states passed the symmetry check in the initial space." << std::endl;
            subspace = subspace_file; 

            if(subspace.empty()){
                std::cerr << "Error: initial subspace from file is empty." << std::endl; std::exit(1);
            }
        H_subspace = get_H_subspace(subspace_file);
        Spectra::SparseGenMatProd<double> op(H_subspace);
        Spectra::SymEigsSolver<Spectra::SparseGenMatProd<double>> eigs(op, 1, min(64, (int)subspace_file.size()));
        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 10000, 1e-10);
        Eigen::VectorXd eigenvalues;
        if(eigs.info() == Spectra::CompInfo::Successful){
            Eigen::VectorXd eigenvalues = eigs.eigenvalues();
                std::cout << "Ground state energy: " << eigenvalues(0) + mol.e_core << std::endl;
                std::cout << "Iterations used: " << eigs.num_iterations() << std::endl;
                std::cout << "Matrix operations: " << eigs.num_operations() << std::endl;

                Eigen::VectorXd evals = eigs.eigenvalues();
                Eigen::MatrixXd evecs = eigs.eigenvectors();
                Eigen::VectorXd v = evecs.col(0);
                Eigen::VectorXd r = H_subspace * v - evals(0) * v;
                std::cout << "True residual[" << 0 << "] = " << r.norm() << std::endl;

        }
        else{
            std::cerr << "Energy computation failed!" << std::endl;
        }

    }

    if(args.load_checkpoint != ""){
        string _tmpstr;
        Eigen::MatrixXd subspace_vec;
        Eigen::VectorXcd params_vec;
        _tmpstr = load_checkpoint(args.load_checkpoint, params_vec, subspace_vec);

        tie(params_NN.a, params_NN.W, params_NN.b) = unflatten(params_NN, params_vec.array());
        subspace.clear();
        for(int r = 0; r < subspace_vec.rows(); r++){
            subspace.push_back(vec2slater(subspace_vec.row(r)));
            assert(slater2vec(vec2slater(subspace_vec.row(r)),0,mol.nCasOrb) == subspace_vec.row(r));
        }
        cout << "Loaded checkpoint from " << args.load_checkpoint << endl;
    }else{
        cout << "Checkpoint not given, starting from random parameters" << endl;
    }

    H_subspace = get_H_subspace(subspace);
    cout << "Initial subspace size: " << subspace.size() << endl;
    assert(subspace.size() > 0);
    
    cout <<"System Time: "<< strtime() <<endl;

    cout << " ====== OPTIMIZATION START ====== "<<endl;

    double best_energy = 0;

    for(int step=0; step<args.max_iter+1; step++)
    {
        int macro_step = step / args.subspace_iter;
        int micro_step = step % args.subspace_iter;
        const double active_guide_weight =
            (args.guide_stop_macro >= 0 && macro_step >= args.guide_stop_macro)
            ? 0.0 : args.guide_weight;

        cout << strtime();
        printf("Step %d.%d start ...\n", macro_step, micro_step);
        cout << "  Subspace Size: " << subspace.size() <<endl;

        Eigen::VectorXcd logwfn = NN_forward_loop(params_NN, subspace, mol.nCasOrb);

        double max_wfn_real = logwfn.real().maxCoeff();
        logwfn = logwfn.array() - logwfn.real().maxCoeff();
        Eigen::VectorXcd wfn = logwfn.array().exp();

        assert(!wfn.array().isInf().any());
        assert(!wfn.array().isNaN().any());

        Eigen::VectorXd rho = wfn.cwiseAbs2();
        rho.array() /= rho.sum();
        assert(rho.allFinite());

        if(args.init_sample == "file"){
            unordered_map<SlaterInt_t, double, SlaterIntHash> rho_map;
            for(size_t i=0; i<subspace.size(); i++){
                rho_map[subspace[i]] = rho[i];
            }

            double init_prob_sum = 0;
            size_t init_prob_count = 0;
            for(auto &it: subspace_file){
                if(rho_map.find(it) != rho_map.end()){
                    init_prob_sum += rho_map[it];
                    init_prob_count++;
                }
            }
            printf(" %zu of %zu init samples found in current subspace, prob sum: %.3f \n", init_prob_count, subspace_file.size(), init_prob_sum);
        }

        Eigen::VectorXcd elocs;
        Eigen::dcomplex eng;
        if(micro_step==0 && step>args.mcmc_iter){

            WfnMap_t selected_subspace_logwfn;
            WfnMap_t selected_subspace_elocs;
            cout <<"  "<< strtime() << "Full SD space Eloc & SC ..." << endl;
            tie(elocs, selected_subspace_logwfn) = omp_eloc(subspace, params_NN, args.n_sample, max_wfn_real + log(args.sample_thresh), true);
            if(selected_subspace_logwfn.size() < args.n_sample){
                args.sample_thresh *= 0.5;
                cout <<"  "<< strtime() << "Reduced sample threshold to " << args.sample_thresh << endl;
            }
            
            if(args.init_sample == "file"){
                Eigen::VectorXcd subspace_file_logwfn = NN_forward_loop(params_NN, subspace_file, mol.nCasOrb);
                for(int i=0; i<subspace_file.size(); i++){
                    if(selected_subspace_logwfn.find(subspace_file[i]) == selected_subspace_logwfn.end()){
                        selected_subspace_logwfn[subspace_file[i]] = subspace_file_logwfn[i];
                    }
                }
            }
            for(int i=0; i<subspace.size(); i++){
                selected_subspace_elocs[subspace[i]] = elocs[i];
            }

            subspace.clear();
            for(auto &it: selected_subspace_logwfn){
                subspace.push_back(it.first);
            }
            H_subspace = get_H_subspace(subspace);
            
            logwfn.resize(selected_subspace_logwfn.size());
            for(int i=0; i<subspace.size(); i++){
                logwfn[i] = selected_subspace_logwfn[subspace[i]];
            }
            logwfn = logwfn.array() - logwfn.real().maxCoeff();
            wfn = logwfn.array().exp();
            rho = wfn.cwiseAbs2();
            rho.array() /= rho.sum();
            assert(rho.allFinite());

            vector<SlaterInt_t> new_minus_old_subspace;
            for(int i=0; i<subspace.size(); i++){
                if(selected_subspace_elocs.find(subspace[i]) == selected_subspace_elocs.end()){
                    new_minus_old_subspace.push_back(subspace[i]);
                }
            }
            elocs = omp_eloc(new_minus_old_subspace, params_NN, args.n_sample,args.sample_thresh,false).first;
            for(int i=0; i<new_minus_old_subspace.size(); i++){
                selected_subspace_elocs[new_minus_old_subspace[i]] = elocs[i];
            }

            elocs.resize(subspace.size());
            for(int i=0; i<subspace.size(); i++){
                elocs[i] = selected_subspace_elocs[subspace[i]];
            }
            assert(elocs.allFinite());

            eng = (elocs.transpose()*rho).sum();

        }else if(step<args.mcmc_iter){
            cout <<"  "<< strtime() << "MCMC & Eloc ..." << endl;
            int n_chain = args.n_thread * 10;
            int n_step = args.n_sample / n_chain + 300;
            subspace = mcmc_sampling(n_chain, n_step, 2*mol.nCasOrb, 10*mol.nCasOrb, params_NN);
            if(args.init_sample == "file"){
                unordered_set<SlaterInt_t, SlaterIntHash> subspace_set;
                for(auto &it: subspace){
                    subspace_set.insert(it);
                }
                for(auto &it: subspace_file){
                    subspace_set.insert(it);
                }
                subspace.clear();
                for(auto &it: subspace_set){
                    subspace.push_back(it);
                }
            }

            H_subspace = get_H_subspace(subspace);

            logwfn = NN_forward_loop(params_NN, subspace, mol.nCasOrb);

            double max_wfn_real = logwfn.real().maxCoeff();
            logwfn = logwfn.array() - logwfn.real().maxCoeff();
            wfn = logwfn.array().exp();
            rho = wfn.cwiseAbs2();
            rho.array() /= rho.sum();

            tie(eng, elocs) = get_energy_subspace_new(H_subspace, wfn, rho);

        }else{
            cout <<"  "<< strtime() << "Subspace Eloc ..." << endl;
            tie(eng, elocs) = get_energy_subspace_new(H_subspace, wfn, rho);
            assert(elocs.allFinite());
        }

        double guide_fidelity = 0.0;
        double physical_eng = eng.real();
        double objective_eng = physical_eng;
        if(active_guide_weight > 0.0) {
            Eigen::VectorXcd guide_elocs;
            tie(guide_fidelity, guide_elocs) = guide_projector_local_energy(subspace, wfn, guide_ci);

            const double safe_fidelity = max(guide_fidelity, 1.0e-14);
            const double guide_force_scale = active_guide_weight / safe_fidelity;
            const double guide_loss = -active_guide_weight * log(safe_fidelity);

            elocs += guide_force_scale * guide_elocs;
            eng += guide_force_scale * (1.0 - guide_fidelity);
            objective_eng += guide_loss;
            printf("  HCI guide fidelity: %.10f, physical energy: %.10f, -lambda*log(F): %.10f\n",
                                 guide_fidelity, physical_eng + mol.e_core, guide_loss);
        }

        cout <<"  "<< strtime() << "SR optim ..." << endl;
        Eigen::VectorXcd dx = sr(params_NN, subspace, rho, elocs, mol.nCasOrb, args.diag_eps, eng, args.n_block);

        cout <<"  "<< strtime() << "Line search ..." << endl;
        vector<double> lr_options = get_lr_options(args.max_lr, args.min_lr, args.n_search);
        double psi_old_psi_old = wfn.dot(wfn).real();
        double min_eng = objective_eng;
        int argmin_eng = -1;

        for(int l=0;l<lr_options.size();l++){
            double lr_test = lr_options[l];

            update_params(params_NN, -lr_test * dx);
            Eigen::VectorXcd logwfn_new = NN_forward_loop(params_NN, subspace, mol.nCasOrb);

            logwfn_new = logwfn_new.array() - logwfn_new.array().real().maxCoeff();
            Eigen::VectorXcd wfn_new = logwfn_new.array().exp();
            assert(!wfn_new.array().isInf().any());
            assert(!wfn_new.array().isNaN().any());
            
            double psi_new_psi_new = wfn_new.dot(wfn_new).real();
            Eigen::dcomplex psi_old_psi_new = wfn.dot(wfn_new);
            double ovlp2 = (psi_old_psi_new * conj(psi_old_psi_new)).real() / (psi_old_psi_old * psi_new_psi_new);
            if(!(ovlp2 > - 1e-12 && ovlp2 < 1.0 + 1e-12)){
                throw std::runtime_error("Numerical Error: Cauchy-Schwarz inequality violated");
            }

            if(ovlp2 < 0.98*0.98){
                update_params(params_NN, lr_test * dx);
                continue;
            }

            Eigen::VectorXd rho_new = wfn_new.cwiseAbs2();
            rho_new.array() /= rho_new.sum();
            double eng_new = get_energy_subspace_new(H_subspace, wfn_new, rho_new).first.real();
            if(active_guide_weight > 0.0) {
                const double fidelity_new = guide_projector_local_energy(subspace, wfn_new, guide_ci).first;
                eng_new += -active_guide_weight * log(max(fidelity_new, 1.0e-14));
            }


            update_params(params_NN, lr_test * dx);
            if(eng_new<min_eng){
                min_eng = eng_new;
                argmin_eng = l;
            }
        }

        Eigen::VectorXcd update_vec;
        if(argmin_eng == -1){
            double _lr = *min_element(lr_options.begin(), lr_options.end());
            update_vec = - _lr * dx;
            cout << "    [Dynamic LR] No valid lr found, use " << _lr <<   " instead" << endl;
        }else{
            update_vec = -lr_options[argmin_eng] * dx;
            cout << "    [Dynamic LR] lr = " << lr_options[argmin_eng] << " found" << endl;
        }
        if(update_vec.norm() > 2.0){
            update_vec = update_vec * (2.0 / update_vec.norm());
            cout << "    [Update Norm] Update norm ||eta*dx|| clipped to 2.0" << endl;
        }else{
            cout << "    [Update Norm] ||eta*dx|| = " << update_vec.norm() << endl;
        }
     
        update_params(params_NN, update_vec);

        cout<<strtime();
        if(physical_eng < best_energy && micro_step!=0){
            best_energy = physical_eng;
        }
        printf("  Physical energy = %lf, Best physical energy = %lf", physical_eng+mol.e_core, best_energy+mol.e_core); cout << endl;

        if(args.save_checkpoint != ""){
            save_checkpoint(args.save_checkpoint + to_string(macro_step) + "_" + to_string(micro_step) + ".npz", 
                flatten(params_NN, params_NN.a, params_NN.W, params_NN.b), 
                slater2vec_batch(subspace, 0, mol.nCasOrb), 
                args.original_cmd
            );
        }

    }

    return 0;
}

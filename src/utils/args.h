#include <iostream>
#include <string>
#include <cstring>
#include <Eigen/Dense>
#include <utils/utils.h>

typedef struct {

    std::string mol_info_file = "./mol_info.json";
    std::string mol_itgs_file = "./dump.FciDmp";
    int max_iter = 100;
    int mcmc_iter = 100;
    int seed = 42;
    int n_thread = 1;

    double alpha = 2.0;     
    double inv_temp = 1.0;

    double diag_eps = 1e-5;
    int subspace_iter = 5;
    double max_lr = 0.2;
    double min_lr = 0.0001;
    int n_search = 10;

    std::string init_sample = "mcmc";
    int n_sample = 5000;
    int n_block = 2;
    double sample_thresh = 0.0;  
    std::string init_subspace_file = "";
    std::string guide_ci_file = "";
    double guide_weight = 0.0;
    int guide_stop_macro = -1;

    std::string load_checkpoint = "";
    std::string save_checkpoint = "";
    std::string original_cmd = "";
    std::string group_name = "d2h";

} Args_t;

static Args_t parse_args(int argc, char* argv[])
{
    Args_t args;
    for(int i=0; i<argc; i++){
        if(!strcmp(argv[i], "--mol_info_file")){ args.mol_info_file = argv[i+1]; }
        if(!strcmp(argv[i], "--mol_itgs_file")){ args.mol_itgs_file = argv[i+1]; }
        if(!strcmp(argv[i], "--seed")){ sscanf(argv[i+1], "%d", &args.seed); }
        if(!strcmp(argv[i], "--max_iter")){ sscanf(argv[i+1], "%d", &args.max_iter); }
        if(!strcmp(argv[i], "--mcmc_iter")){ sscanf(argv[i+1], "%d", &args.mcmc_iter); }
        if(!strcmp(argv[i], "--alpha")){ sscanf(argv[i+1], "%lf", &args.alpha); }
        if(!strcmp(argv[i], "--inv_temp")){ sscanf(argv[i+1], "%lf", &args.inv_temp); }
        if(!strcmp(argv[i], "--diag_eps")){ sscanf(argv[i+1], "%lf", &args.diag_eps); }
        if(!strcmp(argv[i], "--subspace_iter")){ sscanf(argv[i+1], "%d", &args.subspace_iter); }
        if(!strcmp(argv[i], "--n_sample")){ sscanf(argv[i+1], "%d", &args.n_sample); }
        if(!strcmp(argv[i], "--n_block")){ sscanf(argv[i+1], "%d", &args.n_block); }
        if(!strcmp(argv[i], "--max_lr")){ sscanf(argv[i+1], "%lf", &args.max_lr); }
        if(!strcmp(argv[i], "--min_lr")){ sscanf(argv[i+1], "%lf", &args.min_lr); }
        if(!strcmp(argv[i], "--n_search")){ sscanf(argv[i+1], "%d", &args.n_search); }
        if(!strcmp(argv[i], "--sample_thresh")){ 
            sscanf(argv[i+1], "%lf", &args.sample_thresh); 
        }
        if(!strcmp(argv[i], "--init_subspace_file")){ args.init_subspace_file = argv[i+1]; }
        if(!strcmp(argv[i], "--guide_ci_file")){ args.guide_ci_file = argv[i+1]; }
        if(!strcmp(argv[i], "--guide_weight")){ sscanf(argv[i+1], "%lf", &args.guide_weight); }
        if(!strcmp(argv[i], "--guide_stop_macro")){ sscanf(argv[i+1], "%d", &args.guide_stop_macro); }

        if(!strcmp(argv[i], "--load_checkpoint")){ args.load_checkpoint = argv[i+1]; }
        if(!strcmp(argv[i], "--save_checkpoint")){ args.save_checkpoint = argv[i+1]; }
        if(!strcmp(argv[i], "--init_sample")){ args.init_sample = argv[i+1]; }
        if(!strcmp(argv[i], "--group_name")){ args.group_name = argv[i+1]; }

    }

    for(int i=0; i<argc; ++i) {
        args.original_cmd += argv[i];
        if(i < argc-1) args.original_cmd += " ";
    }

    Eigen::initParallel();
    int N_CPU = Eigen::nbThreads();
    args.n_thread = N_CPU;     

        std::cout << " ======= HYPER PARAMETERS START ======="<<std::endl;
        printf("[Global] num_threads = %d, seed = %d, max_iter = %d, mcmc_iter = %d\n", N_CPU, args.seed, args.max_iter, args.mcmc_iter);
        printf("[Wavefunction] alpha = %.2g, inv_temp = %.2g\n", args.alpha, args.inv_temp);
        printf("[SR Optimizer] learning rate range = [%.2g, %.2g], n_search = %d, subspace_iter = %d, n_block = %d\n", 
            args.max_lr, args.min_lr, args.n_search, args.subspace_iter, args.n_block);
        
        cout << "[SR Optimizer] Learning rate schedule: ";
        for(auto lr: get_lr_options(args.max_lr, args.min_lr, args.n_search)){
            cout << lr << " ";
        }
        cout << endl;

        printf("[Sampling] SC sampling, sample_thresh = %.2g, max_num_sample = %d\n", args.sample_thresh, args.n_sample);
        printf("[Sampling] init_sample = %s\n", args.init_sample.c_str());
        if(!args.init_subspace_file.empty()) {
            printf("[Sampling] init_subspace_file = %s\n", args.init_subspace_file.c_str());
        }
        if(args.guide_weight > 0.0) {
            printf("[HCI guide] ci_file = %s, projector weight = %.6g, stop macro = %d\n",
                   args.guide_ci_file.c_str(), args.guide_weight, args.guide_stop_macro);
        }

        printf("[Molecule] mol_info_file = %s, mol_itgs_file = %s\n", args.mol_info_file.c_str(), args.mol_itgs_file.c_str());
        if(!args.load_checkpoint.empty()) {
            printf("[Checkpoint] load_checkpoint = %s\n", args.load_checkpoint.c_str());
        }
        if(!args.save_checkpoint.empty()) {
            printf("[Checkpoint] save_checkpoint = %s\n", args.save_checkpoint.c_str());
        }
        if(args.init_sample != "mcmc" && args.init_sample != "random" && 
           args.init_sample != "cisd" && args.init_sample != "file") {
            std::cerr << "Error: Unsupported init_sample method '" 
                      << args.init_sample 
                      << "'. Use 'mcmc', 'random', 'cisd', or 'file'." 
                      << std::endl;
            exit(1);
        }
        if(args.group_name != "none") {
            printf("[Molecule] group_name = %s\n", args.group_name.c_str());
        }
        if(args.guide_weight < 0.0 || (args.guide_weight > 0.0 && args.guide_ci_file.empty())) {
            std::cerr << "Error: --guide_weight must be nonnegative and requires --guide_ci_file when positive." << std::endl;
            exit(1);
        }
        if(args.guide_stop_macro < -1) {
            std::cerr << "Error: --guide_stop_macro must be -1 (never stop) or nonnegative." << std::endl;
            exit(1);
        }
        std::cout << " ======== HYPER PARAMETERS END ========\n"<<std::endl;
    return args;  
}

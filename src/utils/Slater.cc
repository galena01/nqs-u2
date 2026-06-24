#include "Slater.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cstring>

void print_slater(const SlaterInt_t& s, int nSpaOrb) {
    const int total = 2 * nSpaOrb + 3;
    char buffer[total];
    buffer[0] = '[';
    buffer[nSpaOrb + 1] = '|';
    buffer[total - 1] = ']';
    buffer[total] = '\0';

    for (int i = 0; i < nSpaOrb; ++i) {
        const int byte_idx = i / 8;
        const int bit_offset = i % 8;
        buffer[i + 1] = (s.alpha[byte_idx] >> bit_offset) & 0x01 ? '1' : '0';
        buffer[i + nSpaOrb + 2] = (s.beta[byte_idx] >> bit_offset) & 0x01 ? '1' : '0';
    }
    std::cout << buffer;
}

int get_bit(const uint8_t* arr, int pos) {

    const int byte_idx = pos / 8;
    const int bit_offset = pos % 8;

    return (arr[byte_idx] >> bit_offset) & 0x01;
}

void set_bit(uint8_t* arr, int pos, int val) {

    const int byte_idx = pos / 8;
    const int bit_offset = pos % 8;

    const uint8_t mask = 1 << bit_offset;

    if (val) {
        arr[byte_idx] |= mask;
    } else {
        arr[byte_idx] &= ~mask;
    }
}

SlaterInt_t hf_state(int nAlpha, int nBeta)
{

    SlaterInt_t hf = {0};
    for(int i=0;i<nAlpha;i++){
        set_bit(hf.alpha, i, 1);
    }
    for(int i=0;i<nBeta;i++){
        set_bit(hf.beta, i, 1);
    }
    return hf;
}


Eigen::RowVectorXd slater2vec(const SlaterInt_t& s, int cas_low, int cas_high) {
    const int n_orb = cas_high - cas_low;
    Eigen::RowVectorXd vec(2 * n_orb);

    for (int i = 0; i < n_orb; ++i) {
        int global_pos = cas_low + i;

        vec[i] = static_cast<double>(get_bit(s.alpha, global_pos));

        vec[i+n_orb] = static_cast<double>(get_bit(s.beta, global_pos));
    }

    return vec;
}

static int ctz8(uint8_t x) {
    if (x == 0) return 8;
    constexpr int bit_perm[8] = {0, 1, 2, 7, 3, 13, 8, 19};
    return bit_perm[(0x218A392CD3D5DBFULL * (x & -x)) >> 58];
}

void get_occ_index(const uint8_t *arr, std::vector<int> &idx) {
    idx.clear();

    for (int byte_idx = 0; byte_idx < MAX_SLATER_SIZE; ++byte_idx) {
        uint8_t current_byte = arr[byte_idx];
        if (current_byte == 0) continue;

        while (current_byte != 0) {
            int bit_pos = ctz8(current_byte);
            if (bit_pos >= 8) break;

            int global_pos = byte_idx * 8 + bit_pos;
            idx.push_back(global_pos);

            current_byte &= current_byte - 1;
        }
    }

}

void get_vir_index(const uint8_t *arr, int nSpaOrb, std::vector<int> &idx) {
    idx.clear();

    const int total_bytes = nSpaOrb / 8 + (nSpaOrb % 8 ? 1 : 0);
    const int last_byte_bits = nSpaOrb % 8;

    for (int byte_idx = 0; byte_idx < total_bytes; ++byte_idx) {
        uint8_t byte_mask = 0xFF;
        if (byte_idx == total_bytes - 1 && last_byte_bits != 0) {
            byte_mask = (1 << last_byte_bits) - 1;
        }

        const uint8_t vir_byte = (~arr[byte_idx]) & byte_mask;
        uint8_t current = vir_byte;

        while (current != 0) {
            const int bit_pos = ctz8(current);
            if (bit_pos >= 8) break;

            const int global_pos = byte_idx * 8 + bit_pos;
            if (global_pos < nSpaOrb) {
                idx.push_back(global_pos);
            }

            current &= current - 1;
        }
    }
}


SlaterDiffSpa_t diff_index_spa(const uint8_t* arr1, const uint8_t* arr2) {
    SlaterDiffSpa_t diff = {0};
    std::vector<int> p_idx, h_idx;

    p_idx.reserve(8);
    h_idx.reserve(8);

    for(int i=0;i<MAX_SLATER_SIZE*8;i++){
        if(get_bit(arr1,i)!=get_bit(arr2,i)){
            if(get_bit(arr1,i)==1){ p_idx.push_back(i); }
            else if(get_bit(arr1,i)==0){ h_idx.push_back(i); }
            else{assert(0);}
        }
    }

    const size_t n_exc = p_idx.size();
    assert(n_exc==h_idx.size());
    diff.nExc = n_exc;

    if (n_exc > 0) {
        diff.p[0] = p_idx[0];
        diff.h[0] = h_idx[0];
    }
    if (n_exc > 1) {
        diff.p[1] = p_idx[1];
        diff.h[1] = h_idx[1];
    }

    return diff;
}

SlaterDiff_t diff_index_int(const SlaterInt_t& s1, const SlaterInt_t& s2) {
    return {
        diff_index_spa(s1.alpha, s2.alpha),
        diff_index_spa(s1.beta,  s2.beta)
    };
}


int diff_parity_spa(const uint8_t* occ, SlaterDiffSpa_t diff) {
    int cnt = 0;

    auto process_interval = [&](int start, int end) {
        for (int pos = start; pos < end; ++pos) {
            cnt += get_bit(occ, pos);
        }
    };

    if (diff.nExc == 2) {
        const int pmin0 = std::min(diff.p[0], diff.h[0]);
        const int pmax0 = std::max(diff.p[0], diff.h[0]);
        process_interval(pmin0 + 1, pmax0);

        const int pmin1 = std::min(diff.p[1], diff.h[1]);
        const int pmax1 = std::max(diff.p[1], diff.h[1]);
        process_interval(pmin1 + 1, pmax1);

        if ((diff.p[0] > diff.h[1]) || (diff.h[0] > diff.p[1])) {
            cnt++;
        }
    }
    else if (diff.nExc == 1) {
        const int posmin = std::min(diff.p[0], diff.h[0]);
        const int posmax = std::max(diff.p[0], diff.h[0]);
        process_interval(posmin + 1, posmax);
    }

    return (cnt % 2) ? -1 : 1;
}

int diff_parity_int(const SlaterInt_t &s1, const SlaterDiff_t &diff) {
    return diff_parity_spa(s1.alpha, diff.first) *
           diff_parity_spa(s1.beta,  diff.second);
}

uint64_t get_wfn_irrep(SlaterInt_t slater, int nCasOrb, int nAlpha, int nBeta,
                      const string& groupname, const vector<uint64_t> &orbsym)
{
    uint64_t wfnsym = 0;
    vector<int> occ(nCasOrb, 0);

    for (int i = 0; i < nCasOrb; i++) {
        occ[i] = get_bit(slater.alpha, i) + get_bit(slater.beta, i);
    }

    for (int i = 0; i < nCasOrb; i++) {
        if (occ[i] == 1) {
            wfnsym ^= (orbsym[i] % 10);
        }
    }

    if (groupname == "dooh" || groupname == "coov") {

        vector<int> orb_l(nCasOrb, 0);

        for (int i = 0; i < nCasOrb; i++) {
            orb_l[i] = static_cast<int>(orbsym[i] / 10) * 2;
            uint64_t irrep = orbsym[i] % 10;

            if (irrep == 2 || irrep == 3 || irrep == 6 || irrep == 7) {
                orb_l[i] += 1;
            }

            if (irrep == 1 || irrep == 3 || irrep == 4 || irrep == 6) {
                orb_l[i] = -orb_l[i];
            }
        }

        int wfn_momentum = 0;
        for (int i = 0; i < nCasOrb; i++) {
            if (occ[i] != 0) wfn_momentum += orb_l[i];
            if (occ[i] == 2) wfn_momentum += orb_l[i];
        }

        wfnsym += (std::abs(wfn_momentum) / 2) * 10;
    }
    return wfnsym;
}
bool spa_symm_allowed(SlaterInt_t slater, int nCasOrb, int nAlpha, int nBeta,
                     const string& groupname, const vector<uint64_t> &orbsym,
                     const vector<uint64_t> &target_irrep_ids)
{
    if (target_irrep_ids.empty()) {
        return true;
    }

    uint64_t wfnsym = get_wfn_irrep(slater, nCasOrb, nAlpha, nBeta, groupname, orbsym);
    return (std::find(target_irrep_ids.begin(), target_irrep_ids.end(), wfnsym) != target_irrep_ids.end());
}


vector<SlaterInt_t> sd_space(SlaterInt_t slater, const vector<uint64_t> &symm, int nCasOrb, int nAlpha, int nBeta, const string& groupname, const vector<uint64_t> &target_irrep_ids)
{
    vector<SlaterInt_t> sd;

    int particle_list_alpha[MAX_SLATER_SIZE*8]; int pcount_alpha=0;
    int hole_list_alpha[MAX_SLATER_SIZE*8]; int hcount_alpha=0;

    int particle_list_beta[MAX_SLATER_SIZE*8]; int pcount_beta=0;
    int hole_list_beta[MAX_SLATER_SIZE*8]; int hcount_beta=0;


    for(int i=0;i<nCasOrb;i++){
        if(get_bit(slater.alpha,i)==1)    {particle_list_alpha[pcount_alpha] = i; pcount_alpha++;}
        else                              {hole_list_alpha[hcount_alpha]=i; hcount_alpha++;}

        if(get_bit(slater.beta,i)==1)   {particle_list_beta[pcount_beta] = i; pcount_beta++;}
        else                            {hole_list_beta[hcount_beta]=i; hcount_beta++;}
    }

    int row_count=0;

    sd.push_back(slater);

    for(int i=0;i<pcount_alpha;i++){
        for(int j=0;j<hcount_alpha;j++){
            SlaterInt_t new_state = slater;
            set_bit(new_state.alpha,particle_list_alpha[i],0);
            set_bit(new_state.alpha,hole_list_alpha[j],1);
            if(!spa_symm_allowed(new_state, nCasOrb, nAlpha, nBeta, groupname, symm, target_irrep_ids)){continue;}
            sd.push_back(new_state);
        }
    }

    for(int i=0;i<pcount_beta;i++){
        for(int j=0;j<hcount_beta;j++){
            SlaterInt_t new_state = slater;
            set_bit(new_state.beta,particle_list_beta[i],0);
            set_bit(new_state.beta,hole_list_beta[j],1);
            if(!spa_symm_allowed(new_state, nCasOrb, nAlpha, nBeta, groupname, symm, target_irrep_ids)){continue;}
            sd.push_back(new_state);
        }
    }

    for(int i=0;i<pcount_alpha;i++){
        for(int j=0;j<i;j++){
            for(int k=0;k<hcount_alpha;k++){
                for(int l=0;l<k;l++){
                    SlaterInt_t new_state = slater;

                    set_bit(new_state.alpha,particle_list_alpha[i],0);
                    set_bit(new_state.alpha,hole_list_alpha[k],1);
                    set_bit(new_state.alpha,particle_list_alpha[j],0);
                    set_bit(new_state.alpha,hole_list_alpha[l],1);
                    if(!spa_symm_allowed(new_state, nCasOrb, nAlpha, nBeta, groupname, symm, target_irrep_ids)){continue;}
                    sd.push_back(new_state);
                }
            }
        }
    }

    for(int i=0;i<pcount_beta;i++){
        for(int j=0;j<i;j++){
            for(int k=0;k<hcount_beta;k++){
                for(int l=0;l<k;l++){
                    SlaterInt_t new_state = slater;


                    set_bit(new_state.beta,particle_list_beta[i],0);
                    set_bit(new_state.beta,hole_list_beta[k],1);
                    set_bit(new_state.beta,particle_list_beta[j],0);
                    set_bit(new_state.beta,hole_list_beta[l],1);
                    if(!spa_symm_allowed(new_state, nCasOrb, nAlpha, nBeta, groupname, symm, target_irrep_ids)){continue;}
                    sd.push_back(new_state);
                }
            }
        }
    }

    for(int i=0;i<pcount_alpha;i++){
        for(int j=0;j<pcount_beta;j++){
            for(int k=0;k<hcount_alpha;k++){
                for(int l=0;l<hcount_beta;l++){
                    SlaterInt_t new_state = slater;
                    set_bit(new_state.alpha,particle_list_alpha[i],0);
                    set_bit(new_state.alpha,hole_list_alpha[k],1);

                    set_bit(new_state.beta,particle_list_beta[j],0);
                    set_bit(new_state.beta,hole_list_beta[l],1);
                    if(!spa_symm_allowed(new_state, nCasOrb, nAlpha, nBeta, groupname, symm,target_irrep_ids)){continue;}
                    sd.push_back(new_state);
                }
            }
        }
    }


    return sd;

}


Eigen::MatrixXd slater2vec_batch(vector<SlaterInt_t> slater_list, int cas_low, int cas_high)
{
    Eigen::MatrixXd slater_batch(slater_list.size(), 2*(cas_high-cas_low));

    #pragma omp parallel for
    for(int i=0;i<slater_list.size();i++){
        slater_batch.row(i)=slater2vec(slater_list[i],cas_low,cas_high);
    }
    return slater_batch;
}

SlaterInt_t vec2slater(Eigen::RowVectorXd vec){
    int nCasOrb = vec.size()/2;
    assert(vec.size()%2==0);
    assert(nCasOrb<=MAX_SLATER_SIZE*8);
    SlaterInt_t new_state = {0};

    for (int i = 0; i < nCasOrb; ++i) {
        if (vec[i] > 0.5) {
            set_bit(new_state.alpha, i, 1);
        }

        if (vec[i + nCasOrb] > 0.5) {
            set_bit(new_state.beta, i, 1);
        }
    }

    return new_state;
}

void shuffle_bits(uint8_t* arr, int nCasOrb, std::mt19937_64& rng) {
    for (int i = nCasOrb - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        const int j = dist(rng);

        const int bit_i = get_bit(arr, i);
        const int bit_j = get_bit(arr, j);
        set_bit(arr, i, bit_j);
        set_bit(arr, j, bit_i);
    }
}


std::vector<SlaterInt_t> load_subspace_from_file(
    const std::string& filename,
    int nSpaOrb)
{
    std::ifstream fin(filename);
    std::vector<SlaterInt_t> result;
    std::string line;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);

        SlaterInt_t s{};

        for (int i = 0; i < nSpaOrb; ++i) {
            std::string tok;
            iss >> tok;

            if (tok == "2") {
                set_bit(s.alpha, i, 1);
                set_bit(s.beta,  i, 1);
            } else if (tok == "u") {
                set_bit(s.alpha, i, 1);
            } else if (tok == "d") {
                set_bit(s.beta, i, 1);
            } else if (tok == "0") {
            } else {
                std::cerr << "Invalid token: " << tok << std::endl;
                exit(1);
            }
        }

        result.push_back(s);
    }

    return result;
}

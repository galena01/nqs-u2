#pragma once

#include "Eigen/Core"
#include <vector>
#include <string>
#include <iostream>
#include <random>

#define MAX_SLATER_SIZE 6

using namespace std;

typedef Eigen::RowVectorXd Slater_t;
typedef Eigen::MatrixXd SlaterList_t;

typedef struct{
    uint8_t alpha[MAX_SLATER_SIZE];
    uint8_t beta[MAX_SLATER_SIZE];
}SlaterInt_t;

typedef enum{ALPHA, BETA}Spin_t;

typedef struct{
    int nExc;
    int p[2];
    int h[2];
}SlaterDiffSpa_t;

typedef pair<SlaterDiffSpa_t,SlaterDiffSpa_t> SlaterDiff_t;


static bool operator==(const SlaterInt_t& lhs, const SlaterInt_t& rhs) {
    return std::memcmp(lhs.alpha, rhs.alpha, MAX_SLATER_SIZE) == 0 &&
           std::memcmp(lhs.beta, rhs.beta, MAX_SLATER_SIZE) == 0;
}


struct SlaterIntHash {
    std::size_t operator()(const SlaterInt_t& slater) const noexcept {
        constexpr std::size_t prime = 0x9e3779b1;
        std::size_t h = 0;

        for (std::size_t i = 0; i < MAX_SLATER_SIZE; ++i) {
            h ^= static_cast<std::size_t>(slater.alpha[i]) + prime + (h << 6) + (h >> 2);
        }
        for (std::size_t i = 0; i < MAX_SLATER_SIZE; ++i) {
            h ^= static_cast<std::size_t>(slater.beta[i]) + prime + (h << 6) + (h >> 2);
        }
        return h;
    }
};

void print_slater(const SlaterInt_t& s, int nSpaOrb);

int get_bit(const uint8_t* arr, int pos);

void set_bit(uint8_t* arr, int pos, int val);

SlaterInt_t hf_state(int nAlpha, int nBeta);

Eigen::RowVectorXd slater2vec(const SlaterInt_t& s, int cas_low, int cas_high);

void get_occ_index(const uint8_t *arr, std::vector<int> &idx);

void get_vir_index(const uint8_t *arr, int nSpaOrb, std::vector<int> &idx);

SlaterDiff_t diff_index_int(const SlaterInt_t& s1, const SlaterInt_t& s2);

int diff_parity_spa(const uint8_t* occ, SlaterDiffSpa_t diff);

int diff_parity_int(const SlaterInt_t &s1, const SlaterDiff_t &diff);

bool spa_symm_allowed(SlaterInt_t slater, int nCasOrb, int nAlpha, int nBeta, const string& groupname, const vector<uint64_t> &orbsym, const vector<uint64_t> &target_irrep_ids);

vector<SlaterInt_t> sd_space(SlaterInt_t slater, const vector<uint64_t> &symm, int nCasOrb, int nAlpha, int nBeta, const string& groupname, const vector<uint64_t> &target_irrep_ids);

Eigen::MatrixXd slater2vec_batch(vector<SlaterInt_t> slater_list, int cas_low, int cas_high);

SlaterInt_t vec2slater(Eigen::RowVectorXd vec);

uint64_t get_wfn_irrep(SlaterInt_t slater, int nCasOrb, int nAlpha, int nBeta,
                      const string& groupname, const vector<uint64_t> &orbsym) ;

void shuffle_bits(uint8_t* arr, int nBits, std::mt19937_64& rng);

std::vector<SlaterInt_t> load_subspace_from_file(
    const std::string& filename,
    int nSpaOrb);

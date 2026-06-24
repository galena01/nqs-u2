#pragma once

#include <ctime>
#include <string>
#include <map>
#include <vector>


extern char time_string[];

char *strtime();

#include <random>

using namespace std;

class MultiThreadRNG
{
    public:

        void init(int n_thread,int seed=0);

        mt19937_64 &engine();

        static MultiThreadRNG instance;
        MultiThreadRNG(const MultiThreadRNG &) = delete;

        MultiThreadRNG &operator=(const MultiThreadRNG &) = delete;

        MultiThreadRNG() = default;
        ~MultiThreadRNG() = default;

    private:
        vector<mt19937_64> engines;
};

extern MultiThreadRNG oRNG;

vector<double> get_lr_options(double max_lr, double min_lr, int n_search);

static const std::map<std::string, std::vector<std::string>> pyscf_irrep_map = {
    {"base", {"0"}},
    {"c1", {"a"}},
    {"c2v", {"a1", "a2", "b1", "b2"}},
    {"d2h", {"ag", "b1g", "b2g", "b3g", "au", "b1u", "b2u", "b3u"}}
};

static const std::map<std::string, std::vector<std::string>> molcas_irrep_map = {
    {"base", {"1"}},
    {"c1", {"a"}},
    {"c2v", {"a1", "b1", "a2", "b2"}},
    {"d2h", {"ag", "b3u", "b2u", "b1g", "b1u", "b2g", "b3g", "au"}}
};

int irrep2id(const std::string& group_name, const std::string& symbol,
             const std::map<std::string, std::vector<std::string>>& irrep_map);

std::string id2irrep(const std::string& group_name, int irrep_id,
                     const std::map<std::string, std::vector<std::string>>& irrep_map);

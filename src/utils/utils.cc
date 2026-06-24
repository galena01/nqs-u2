#include <omp.h>
#include "utils.h"
char time_string[50];
#include <stdexcept>
#include <stdio.h>

char *strtime()
{

    time_t secs;
    time(&secs);
    char *time_str=ctime(&secs);
    time_string[0]='[';
    int i=1;
    for(;;i++){
        time_string[i]=time_str[i-1];
        if(time_str[i]=='\n'){break;}
    }
    i++;
    time_string[i]=']';
    time_string[i+1]=' ';
    time_string[i+2]=0;

    return time_string;
}



MultiThreadRNG oRNG;

void MultiThreadRNG::init(int n_thread,int seed)
{
    minstd_rand0 e0(seed);
    uniform_int_distribution<> uniform_int(0,1<<30);
    for(int i=0;i<n_thread;i++)
    {
        int seed_i=uniform_int(e0);
        engines.push_back(mt19937_64(seed_i));
    }
}

mt19937_64 &MultiThreadRNG::engine()
{
    return engines[omp_get_thread_num()];
}

vector<double> get_lr_options(double max_lr, double min_lr, int n_search){
    vector<double> res;
    if (n_search <= 0 || max_lr <= min_lr || max_lr < 0 || min_lr < 0) {
        throw invalid_argument("Invalid arguments");
    }

    if (n_search == 1) {
        res.push_back(max_lr);
        return res;
    }

    double log_max = log(max_lr);
    double log_min = log(min_lr);
    double delta = log_max - log_min;
    double step = delta / (n_search - 1);

    for (int i = 0; i < n_search; ++i) {
        double log_val = log_max - i * step;
        res.push_back(exp(log_val));
    }

    return res;
}


#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <stdexcept>

int get_base(const std::map<std::string, std::vector<std::string>>& irrep_map) {
    auto it = irrep_map.find("base");
    if (it == irrep_map.end() || it->second.empty()) {
        throw std::invalid_argument("irrep_map must contain a 'base' key with a valid value");
    }
    try {
        return std::stoi(it->second[0]);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid base value in irrep_map: " + it->second[0]);
    }
}

std::string id2irrep(const std::string& group_name, int irrep_id,
                     const std::map<std::string, std::vector<std::string>>& irrep_map) {
    int base = get_base(irrep_map);

    auto it = irrep_map.find(group_name);
    if (it == irrep_map.end()) {
        throw std::invalid_argument("Unsupported point group: " + group_name);
    }

    if (irrep_id < base || static_cast<size_t>(irrep_id - base) >= it->second.size()) {
        throw std::invalid_argument("Invalid irrep_id: " + std::to_string(irrep_id) +
                                    " for group " + group_name + " (base=" + std::to_string(base) + ")");
    }

    return it->second[irrep_id - base];
}

int irrep2id(const std::string& group_name, const std::string& symbol,
             const std::map<std::string, std::vector<std::string>>& irrep_map) {
    int base = get_base(irrep_map);

    auto it = irrep_map.find(group_name);
    if (it == irrep_map.end()) {
        throw std::invalid_argument("Unsupported point group: " + group_name);
    }

    auto pos = std::find(it->second.begin(), it->second.end(), symbol);
    if (pos == it->second.end()) {
        throw std::invalid_argument("Invalid symbol: " + symbol + " for group " + group_name);
    }

    return static_cast<int>(std::distance(it->second.begin(), pos)) + base;
}


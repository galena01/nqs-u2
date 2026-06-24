#pragma once

#define INTEGRAL_THRESH 1e-12

#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class Molecule
{
    private:
        Molecule() = default;

        Molecule(const Molecule&) = delete;
        Molecule& operator=(const Molecule&) = delete;

        static Molecule instance;

    public:
        string structure;
        string unit;
        string basis_set;

        int mutiplicity;
        int nCasOrb;
        int nCasEle;
        int nCasAlpha, nCasBeta;

        double e_core;
        double e_mf;

        vector<uint64_t> irrep_ids;
        vector<uint64_t> target_irrep_ids;
        string groupname;
        vector<double> mo_energy;

        vector<double> h1e;
        unordered_map<uint64_t, double> g2e;

        double get_g2e(int i, int j, int k, int l);

        int load_fcidump_molcas(string filename, string args_groupname);

        static Molecule& get_instance() {
            return instance;
        }
};

std::ostream& operator<<(std::ostream& os, const Molecule& mol);

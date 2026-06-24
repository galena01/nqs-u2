
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <map>
#include <algorithm>
#include <regex>

#include "utils/utils.h"
#include "MolLoader.h"

using namespace std;

Molecule Molecule::instance;

inline uint64_t pack_key(int i, int j, int k, int l) {
    return (uint64_t(i) << 48)
         | (uint64_t(j) << 32)
         | (uint64_t(k) << 16)
         |  uint64_t(l);
}

int Molecule::load_fcidump_molcas(std::string filename, string args_groupname) {
    groupname = args_groupname;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cout << "Can't open file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    std::string header;
    while (std::getline(infile, line)) {
        header += line + "\n";
        if (line.find("&END") != std::string::npos) {
            break;
        }
    }

    int ms2 = 0;
    int isym = 0;
    int norb = 0;
    int nelec = 0;
    irrep_ids.clear();

    std::smatch match;
    std::istringstream header_iss(header);

    std::regex norb_regex(R"(NORB\s*=\s*(\d+))");
    std::regex nelec_regex(R"(NELEC\s*=\s*(\d+))");
    std::regex ms2_regex(R"(MS2\s*=\s*([-\d]+))");
    std::regex isym_regex(R"(ISYM\s*=\s*(\d+))");
    std::regex orbsym_regex(R"(ORBSYM\s*=\s*([\d\s,]+))");

    while (std::getline(header_iss, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        if (std::regex_search(line, match, norb_regex)) {
            norb = std::stoi(match[1]);
        }
        if (std::regex_search(line, match, nelec_regex)) {
            nelec = std::stoi(match[1]);
        }
        if (std::regex_search(line, match, ms2_regex)) {
            ms2 = std::stoi(match[1]);
        }
        if (std::regex_search(line, match, isym_regex)) {
            isym = std::stoi(match[1]);
        }
        if (std::regex_search(line, match, orbsym_regex)) {
            std::string sym_str = match[1].str();
            sym_str.erase(std::remove_if(sym_str.begin(), sym_str.end(), ::isspace), sym_str.end());
            std::istringstream iss(sym_str);
            std::string num;
            while (std::getline(iss, num, ',')) {
                if (!num.empty()) {
                    irrep_ids.push_back(std::stoi(num));
                }
            }
        }
    }

    string irrep_symbol; int pyscf_id;
    for (auto& id : irrep_ids) {
        irrep_symbol = id2irrep(groupname, id, molcas_irrep_map);
        pyscf_id = irrep2id(groupname, irrep_symbol, pyscf_irrep_map);
        id = pyscf_id;
    }

    nCasOrb = norb;
    nCasEle = nelec;
    mutiplicity = std::abs(ms2) + 1;
    nCasAlpha = (nCasEle + ms2) / 2;
    nCasBeta = (nCasEle - ms2) / 2;

    irrep_symbol = id2irrep(groupname, isym+1, molcas_irrep_map);
    pyscf_id = irrep2id(groupname, irrep_symbol, pyscf_irrep_map);
    target_irrep_ids = {static_cast<uint64_t>(pyscf_id)};
    structure = "";
    unit = "";
    basis_set = "";
    e_mf = 0.0;

    mo_energy.resize(nCasOrb, 0.0);
    h1e.resize(nCasOrb * nCasOrb, 0.0);
    g2e.clear();

    double value;
    int i, j, k, l;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        if (!(iss >> value >> i >> j >> k >> l)) continue;
        if (std::abs(value) < 1e-12) continue;

        if (i == 0 && j == 0 && k == 0 && l == 0) {
            e_core = value;
            continue;
        } else if (k == 0 && l == 0) {
            if (j == 0) {
                mo_energy[i - 1] = value;
            } else {
                h1e[(i - 1) * nCasOrb + (j - 1)] = value;
                if (i != j) {
                    h1e[(j - 1) * nCasOrb + (i - 1)] = value;
                }
            }
        } else {
            i--; j--; k--; l--;
            if (i < j) std::swap(i, j);
            if (k < l) std::swap(k, l);
            if (pack_key(i, j, k, l) < pack_key(k, l, i, j)) {
                std::swap(i, k);
                std::swap(j, l);
            }
            uint64_t idx = pack_key(i, j, k, l);
            if (g2e.find(idx) != g2e.end()) {
                assert(std::abs(g2e[idx] - value) < 1e-12);
            } else {
                g2e[idx] = value;
            }
        }
    }

    infile.close();
    return 0;
}

std::ostream& operator<<(std::ostream& os, const Molecule& mol) {
    os << " ======= MOLECULE INFO START =======\n";
    os << "  Structure: " << mol.structure << "\n";
    os << "  Unit: " << mol.unit << "\n";
    os << "  Basis Set: " << mol.basis_set << "\n";
    os << "  Multiplicity: " << mol.mutiplicity << "\n";
    os << "  nCasOrb: " << mol.nCasOrb << "\n";
    os << "  nCasEle: " << mol.nCasEle << " (" << mol.nCasAlpha << " alpha, " << mol.nCasBeta << " beta)\n";
    os << "  e_core: " << mol.e_core << "\n";
    os << "  e_mf: " << mol.e_mf << "\n";
    os << "  Group Name: " << mol.groupname << "\n";

    os << "  Irrep IDs: ";
    for (auto id : mol.irrep_ids) {
        os << id << " ";
    }
    os << "\n";

    os << "  MO Energies: ";
    for (auto energy : mol.mo_energy) {
        os << energy << " ";
    }
    os << "\n";

    os << "  Target Irrep IDs: ";
    for (auto id : mol.target_irrep_ids) {
        os << id << " ";
    }

    os << "\n";

    os << "  Integral Stats:\n";
    os << "    h1e size: " << mol.h1e.size() << "\n";
    os << "    g2e size: " << mol.g2e.size() << "\n";
    os << " ======= MOLECULE INFO END =======\n";

    return os;
}


double Molecule::get_g2e(int i, int j, int k, int l) {
    if (i < j) std::swap(i, j);
    if (k < l) std::swap(k, l);
    if (pack_key(i, j, k, l) < pack_key(k, l, i, j)) {
        std::swap(i, k);
        std::swap(j, l);
    }

    size_t idx = pack_key(i, j, k, l);
    auto it = g2e.find(idx);
    return (it != g2e.end()) ? it->second : 0.0;
}

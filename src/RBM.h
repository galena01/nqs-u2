#pragma once

#include "utils/Slater.h"
#include "utils/utils.h"
#include <cstring>

using namespace std;

class NN_Params
{
public:
    Eigen::RowVectorXcd a;
    Eigen::MatrixXcd W;
    Eigen::RowVectorXcd b;
    int vnum;

    NN_Params(int input_dim, double alpha, const vector<double>& mo_eng, double inv_temp);

    NN_Params(const NN_Params& other){
        a = other.a;
        W = other.W;
        b = other.b;
        vnum = other.vnum;
    }

private:
    void init_params(int input_dim, double alpha, const vector<double>& mo_eng, double inv_temp);
};

Eigen::ArrayXcd flatten(const NN_Params& params, const Eigen::RowVectorXcd &a, const Eigen::MatrixXcd &W, const Eigen::RowVectorXcd &b);

std::tuple<Eigen::RowVectorXcd, Eigen::MatrixXcd, Eigen::RowVectorXcd> unflatten(const NN_Params& p, const Eigen::ArrayXcd& flat);

void update_params(NN_Params &p, const Eigen::ArrayXcd &dx);

Eigen::VectorXcd NN_forward_batch(const NN_Params &p, const Eigen::MatrixXd &input);

Eigen::VectorXcd NN_forward_batch_eloc(const NN_Params &p, const vector<SlaterInt_t> &sd, int nCasOrb);

Eigen::VectorXcd NN_forward_loop(const NN_Params &p, const vector<SlaterInt_t> &states, int nCasOrb);

Eigen::ArrayXcd NN_grad(NN_Params &p, Eigen::RowVectorXd x0);

void grad_check();

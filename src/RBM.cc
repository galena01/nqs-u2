#include "RBM.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "MolLoader.h"

using namespace std;


std::complex<double> softplus_c(const std::complex<double>& z) {
    double a = z.real();
    if (a > 0) {
        return z + std::log(std::complex<double>(1.0) + std::exp(-z));
    } else {
        return std::log(std::complex<double>(1.0) + std::exp(z));
    }
}

std::complex<double> sigmoid_c(const std::complex<double>& z) {
    double a = z.real();
    if (a >= 0) {
        return std::complex<double>(1.0) / (std::complex<double>(1.0) + std::exp(-z));
    } else {
        auto ez = std::exp(z);
        return ez / (std::complex<double>(1.0) + ez);
    }
}

NN_Params::NN_Params(int input_dim, double alpha, const vector<double>& mo_eng, double inv_temp) {
    init_params(input_dim, alpha, mo_eng, inv_temp);
}


void NN_Params::init_params(int input_dim, double alpha, const vector<double>& mo_eng, double inv_temp) {
    int nHidden = (int)(alpha*input_dim);

    a = Eigen::RowVectorXcd::Zero(input_dim);

    W = Eigen::MatrixXcd::Zero(input_dim, nHidden);

    b = Eigen::RowVectorXcd::Zero(nHidden);

    std::uniform_real_distribution<> unireal(-0.05, 0.05);

    for (int i = 0; i < input_dim; i++) {
        a(i) = unireal(oRNG.engine()) + unireal(oRNG.engine()) * 1.0i;
    }

    for (int i = 0; i < input_dim; i++) {
        for (int j = 0; j < nHidden; j++) {
            W(i,j) = unireal(oRNG.engine()) + unireal(oRNG.engine()) * 1.0i;
        }
    }

    for (int i = 0; i < nHidden; i++) {
        b(i) = unireal(oRNG.engine()) + unireal(oRNG.engine()) * 1.0i;
    }

    cout << "Inverse temperature for RBM init = " << inv_temp << endl;
    if(inv_temp > 0.0) {
        for(int i = 0; i < input_dim/2; i++) {
            double orb_eng = mo_eng[i];
            a(i) = -inv_temp * orb_eng;
            a(i + input_dim/2) = -inv_temp * orb_eng;
        }
    }

    vnum = (a.size() + W.size() + b.size()) ;
}

Eigen::ArrayXcd flatten(const NN_Params& p, const Eigen::RowVectorXcd &a, const Eigen::MatrixXcd &W, const Eigen::RowVectorXcd &b){
    int a_size = a.size();
    int W_size = W.size();
    int b_size = b.size();

    Eigen::ArrayXcd result(a_size + W_size + b_size);

    for (int i = 0; i < a_size; i++) {
        result[i] = a[i];
    }

    int offset = a_size;
    for (int i = 0; i < W_size; i++) {
        int a = i / W.cols();
        int b = i % W.cols();
        result[offset + i] = W(a,b);
    }

    offset += W_size;
    for (int i = 0; i < b_size; i++) {
        result[offset + i] = b[i];
    }

    return result;
}

std::tuple<Eigen::RowVectorXcd, Eigen::MatrixXcd, Eigen::RowVectorXcd> unflatten(const NN_Params& p, const Eigen::ArrayXcd& flat){
    int a_size = p.a.size();
    int W_rows = p.W.rows();
    int W_cols = p.W.cols();
    int b_size = p.b.size();

    Eigen::RowVectorXcd a(a_size);
    Eigen::MatrixXcd W(W_rows, W_cols);
    Eigen::RowVectorXcd b(b_size);

    int offset = 0;

    for (int i = 0; i < a_size; i++) {
        a(i) = flat[offset + i];
    }
    offset += a_size;

    for (int i = 0; i < W_rows * W_cols; i++) {
        int row = i / W_cols;
        int col = i % W_cols;
        W(row, col) = flat[offset + i];
    }
    offset += W_rows * W_cols;

    for (int i = 0; i < b_size; i++) {
        b(i) = flat[offset + i];
    }

    return std::make_tuple(a, W, b);
}

void update_params(NN_Params &p, const Eigen::ArrayXcd &dx){
    assert(dx.size() == p.vnum);
    Eigen::RowVectorXcd dx_a, dx_b;
    Eigen::MatrixXcd dx_W;

    std::tie(dx_a, dx_W, dx_b) = unflatten(p, dx);
    p.a = p.a + dx_a;
    p.W = p.W + dx_W;
    p.b = p.b + dx_b;
}


Eigen::VectorXcd NN_forward_batch(const NN_Params &p, const Eigen::MatrixXd &input) {
    Eigen::MatrixXcd hidden = ((input * p.W).rowwise() + p.b).unaryExpr(&softplus_c);
    Eigen::VectorXcd visible = input * p.a.transpose();

    return hidden.rowwise().sum() + visible;
}

Eigen::VectorXcd NN_forward_batch_eloc(const NN_Params &p, const vector<SlaterInt_t> &sd, int nCasOrb) {
    Eigen::VectorXcd rtn(sd.size());

    const Eigen::RowVectorXcd &a = p.a;
    const Eigen::RowVectorXcd &b = p.b;
    const Eigen::MatrixXcd &W = p.W;

    Eigen::RowVectorXd ref_state = slater2vec(sd[0], 0, nCasOrb);
    Eigen::dcomplex ref_bia = (a * ref_state.transpose()).sum();
    Eigen::RowVectorXcd ref_ker = (ref_state * W) + b;
    rtn[0] = ref_bia + ref_ker.unaryExpr(&softplus_c).sum();

    for(int i=1; i < sd.size(); i++)
    {
        SlaterDiff_t diff = diff_index_int(sd[0], sd[i]);
        Eigen::dcomplex bia = ref_bia;
        Eigen::RowVectorXcd ker = ref_ker;


        if(diff.first.nExc == 1){
            int o = (diff.first.p[0]);
            int v = (diff.first.h[0]);
            bia = bia - a[o] + a[v];
            ker = ker - W.row(o) + W.row(v);
        }
        if(diff.second.nExc == 1){
            int o = (diff.second.p[0])+nCasOrb;
            int v = (diff.second.h[0])+nCasOrb;
            bia = bia - a[o] + a[v];
            ker = ker - W.row(o) + W.row(v);
        }
        if(diff.first.nExc == 2){
            int o1 = (diff.first.p[0]);
            int v1 = (diff.first.h[0]);
            int o2 = (diff.first.p[1]);
            int v2 = (diff.first.h[1]);
            bia = bia - a[o1] - a[o2] + a[v1] + a[v2];
            ker = ker - W.row(o1) - W.row(o2) + W.row(v1) + W.row(v2);
        }
        if(diff.second.nExc == 2){
            int o1 = (diff.second.p[0])+nCasOrb;
            int v1 = (diff.second.h[0])+nCasOrb;
            int o2 = (diff.second.p[1])+nCasOrb;
            int v2 = (diff.second.h[1])+nCasOrb;
            bia = bia - a[o1] - a[o2] + a[v1] + a[v2];
            ker = ker - W.row(o1) - W.row(o2) + W.row(v1) + W.row(v2);
        }
        rtn[i] = bia + ker.unaryExpr(&softplus_c).sum();
    }

    return rtn;
}

Eigen::VectorXcd NN_forward_loop(const NN_Params &p, const vector<SlaterInt_t> &states, int nCasOrb)
{
    int n_sample = states.size();
    int input_dim = 2*nCasOrb;
    Eigen::VectorXcd result(n_sample);
    int NUM_PER_LOOP = 1280;
    int n_loop = (n_sample - 1)/NUM_PER_LOOP + 1;

    for(int i=0; i < n_loop; i++)
    {
        int start = i * NUM_PER_LOOP;
        int end = min((i+1)*NUM_PER_LOOP, n_sample);
        Eigen::MatrixXd part_states(end-start, input_dim);

        #pragma omp parallel for
        for(int j=0; j < end - start; j++){
            part_states.row(j) = slater2vec(states[start+j], 0, nCasOrb);
        }

        result.block(start,0,end-start,1) = NN_forward_batch(p,part_states);
    }


    return result;
}

Eigen::ArrayXcd NN_grad(NN_Params &p, Eigen::RowVectorXd x0) {
    Eigen::MatrixXcd hidden = ((x0 * p.W).rowwise() + p.b);
    Eigen::VectorXcd visible = x0 * p.a.transpose();

    Eigen::dcomplex y = hidden.unaryExpr(&softplus_c).sum() + visible.value();

    Eigen::MatrixXcd h_sigmoid = hidden.unaryExpr(&sigmoid_c);

    Eigen::ArrayXcd grad = flatten(p, x0, x0.transpose()*h_sigmoid, h_sigmoid);

    return grad;
}

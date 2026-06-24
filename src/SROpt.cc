#include <omp.h>
#include <algorithm>
#include <iostream>

#include <Eigen/Dense>

#include "RBM.h"
#include "utils/Slater.h"
#include "utils/utils.h"
#include "utils/pyutils.h"

using namespace std;

int __step = 0;


void sr_part(NN_Params &params_NN, const vector<SlaterInt_t> &xint, const Eigen::VectorXd &rho, const Eigen::VectorXcd &elocs, int nCasOrb, Eigen::VectorXcd &Omean, Eigen::VectorXcd &OH, vector<Eigen::MatrixXcd> &bOO,int start, int end, const vector<vector<int>> &blk_idxs)
{
    int N_CPU = Eigen::nbThreads();

    Eigen::VectorXcd tOmean[N_CPU];
    Eigen::VectorXcd tOH[N_CPU];

    for(int i=0;i<N_CPU;i++){
        tOH[i].resize(params_NN.vnum);
        tOH[i].setZero();
        tOmean[i].resize(params_NN.vnum);
        tOmean[i].setZero();
    }

    vector<Eigen::MatrixXcd> bOlist;
    for(const vector<int> &idxs: blk_idxs){
        if (idxs.empty() || idxs.size() > params_NN.vnum) {
            throw std::runtime_error("Invalid blk_idxs size");
        }
        bOlist.push_back(Eigen::MatrixXcd::Zero(idxs.size(), end-start));
    }


    #pragma omp parallel for
    for(int i=0;i<end-start;i++){
        int tid = omp_get_thread_num();

        Eigen::RowVectorXd x_i= slater2vec(xint[start+i],0, nCasOrb);
        Eigen::ArrayXcd Oarr(params_NN.vnum);
        Oarr = NN_grad(params_NN,x_i);
        Eigen::VectorXcd O = Oarr;
        double wt=rho[start+i]/rho.sum();

        if (!std::isfinite(wt) || wt < 0) {
            throw std::runtime_error("Invalid wt at sample " + std::to_string(start+i));
        }
        if (!Oarr.allFinite()) {
            throw std::runtime_error("NN_grad returned NaN/Inf at sample " + std::to_string(start+i));
        }

        for(int b=0;b<blk_idxs.size();b++){
            const vector<int> &idxs = blk_idxs[b];
            int n_blk_idx = idxs.size();
            Eigen::VectorXcd O_blk(idxs.size());
            for(int p=0;p<idxs.size();p++){
                O_blk[p] = O[idxs[p]];
                tOmean[tid][idxs[p]] += O[idxs[p]]*wt;
                tOH[tid][idxs[p]] += elocs[start+i] * conj(O[idxs[p]]) * wt;
            }
            bOlist[b].col(i) = (O_blk*sqrt(wt));
        }

    }

    for(int i=0;i<N_CPU;i++){
        for(int j=0;j<2*nCasOrb;j++){
            tOmean[i][j] = tOmean[i][j] / static_cast<double>(blk_idxs.size());
            tOH[i][j] = tOH[i][j] / static_cast<double>(blk_idxs.size());
        }
        OH += tOH[i];
        Omean += tOmean[i];
    }


    #pragma omp parallel for
    for(int b=0;b<blk_idxs.size();b++){
        bOO[b] += (bOlist[b].conjugate() * bOlist[b].transpose());
    }

}

vector<vector<int>> split_block(NN_Params &params_NN, int n_block)
{
    vector<vector<int>> block_indexs;

    int n_hid = params_NN.b.cols();
    int n_vis = params_NN.a.cols();
    assert(params_NN.vnum == n_vis + n_hid + n_vis*n_hid);
    assert(n_hid%n_block==0);
    int n_hid_per_block = n_hid / n_block;

    std::vector<int> idx_hid(n_hid);
    std::iota(idx_hid.begin(), idx_hid.end(), 0);
    std::shuffle(idx_hid.begin(), idx_hid.end(), oRNG.engine());

    std::vector<std::vector<int>> block_idxs;
    block_idxs.reserve(n_block);

    for (int i_blk = 0; i_blk < n_block; ++i_blk) {
        std::vector<int> block_idx;
        block_idx.reserve(n_vis + n_hid_per_block + n_vis * n_hid_per_block);

        for (int i_vis = 0; i_vis < n_vis; ++i_vis) {
            block_idx.push_back(i_vis);
        }

        const int blk_start = i_blk * n_hid_per_block;
        const int blk_end = (i_blk + 1) * n_hid_per_block;
        for (int i = blk_start; i < blk_end; ++i) {
            const int i_hid = idx_hid[i];

            block_idx.push_back(n_vis + n_vis*n_hid + i_hid);

            for (int i_vis = 0; i_vis < n_vis; ++i_vis) {
                block_idx.push_back(n_vis + i_vis * n_hid + i_hid);
            }
        }

        std::sort(block_idx.begin(), block_idx.end());
        block_idxs.push_back(std::move(block_idx));
    }

    return block_idxs;

}

Eigen::VectorXcd sr(NN_Params &params_NN, vector<SlaterInt_t> xint,
                    Eigen::VectorXd rho, Eigen::VectorXcd elocs, int nCasOrb,
                    double eps, Eigen::dcomplex eng, int n_block)
{

    if (!xint.size() || !rho.allFinite() || !elocs.allFinite()) {
        throw std::runtime_error("Invalid input: xint empty or rho/elocs contains NaN/Inf");
    }
    double sum_rho = rho.sum();
    if (sum_rho <= 0 || !std::isfinite(sum_rho)) {
        throw std::runtime_error("rho.sum() is zero, negative, or NaN/Inf");
    }

    vector<vector<int>> blk_idxs;
    blk_idxs = split_block(params_NN, n_block);

    Eigen::VectorXcd Omean(params_NN.vnum);
    Omean.setZero();
    Eigen::VectorXcd OH(params_NN.vnum);
    OH.setZero();
    vector<Eigen::MatrixXcd> bOO;
    for(auto &idx: blk_idxs){
        bOO.push_back(Eigen::MatrixXcd::Zero(idx.size(),idx.size()));
    }

    int SAMPLE_NUM_PER_PART = 2048;
    int n_samples = xint.size();
    int n_part = (n_samples-1)/SAMPLE_NUM_PER_PART + 1;

    cout << "    " << strtime() << "SR optim - Generate matrix elements ..." << endl;

    for(int ip=0;ip<n_part;ip++){
        int start = ip * SAMPLE_NUM_PER_PART;
        int end = min((ip+1)* SAMPLE_NUM_PER_PART, n_samples);
        sr_part(params_NN, xint, rho, elocs, nCasOrb, Omean, OH, bOO, start, end, blk_idxs);
    }



    cout << "    " << strtime() << "SR optim - Solving SR eqn ..." << endl;

    Eigen::VectorXcd dx = Eigen::VectorXcd::Zero(params_NN.vnum);

    for (int b=0; b< n_block; b++) {
        const auto& blk_idx = blk_idxs[b];
        int n_idx_blk = blk_idx.size();

        Eigen::MatrixXcd OO_blk = bOO[b];
        Eigen::VectorXcd Omean_blk(n_idx_blk);
        Eigen::VectorXcd OH_blk(n_idx_blk);

        for (int i = 0; i < n_idx_blk; ++i) {
            OH_blk[i] = OH(blk_idx[i]);
            Omean_blk[i] = Omean(blk_idx[i]);
        }

        if (!OO_blk.allFinite() || !Omean_blk.allFinite() || !OH_blk.allFinite()) {
            throw std::runtime_error("OO_blk, Omean_blk, or OH_blk contains NaN/Inf in block " + std::to_string(b));
        }

        Eigen::MatrixXcd FIM_blk=OO_blk-Omean_blk.conjugate() * Omean_blk.transpose() + eps*Eigen::MatrixXd::Identity(n_idx_blk, n_idx_blk);

        Eigen::VectorXcd grad_blk = OH_blk - eng.real() * Omean_blk.conjugate();
        if (!FIM_blk.allFinite() || !grad_blk.allFinite()) {
            throw std::runtime_error("FIM_blk or grad_blk contains NaN/Inf in block " + std::to_string(b));
        }

        Eigen::VectorXcd dx_sub = scipy_solver(FIM_blk, grad_blk);

        for (int i = 0; i < n_idx_blk; ++i) {
            dx(blk_idx[i]) += dx_sub[i];
        }

    }

    for(int i=0;i<2*nCasOrb;i++){
        dx[i] = dx[i]/static_cast<double>(n_block);
    }

    return dx;
}

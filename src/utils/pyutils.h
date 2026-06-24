#pragma once

#include <Eigen/Core>

Eigen::VectorXcd scipy_solver(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& b);

void save_checkpoint(const std::string& filename,
                    const Eigen::VectorXcd& params,
                    const Eigen::MatrixXd& subspace,
                    const std::string& args);

std::string load_checkpoint(const std::string& filename, Eigen::VectorXcd &params, Eigen::MatrixXd &subspace_vec);

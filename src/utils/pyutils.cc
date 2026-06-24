#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

Eigen::VectorXcd scipy_solver(const Eigen::MatrixXcd& A, const Eigen::VectorXcd& b) {
    py::gil_scoped_acquire acquire;

    try {
        py::module scipy_linalg = py::module::import("scipy.linalg");

        py::array_t<std::complex<double>> A_np = py::cast(A);
        py::array_t<std::complex<double>> b_np = py::cast(b);

        py::array_t<std::complex<double>> x_array = scipy_linalg.attr("solve")(A_np, b_np);

        return Eigen::Map<const Eigen::VectorXcd>(
            x_array.data(),
            x_array.size()
        ).eval();

    } catch (const py::error_already_set& e) {
        std::cerr << "Python error in scipy_solver:\n" << e.what() << std::endl;
        throw std::runtime_error("SciPy solve failed");
    } catch (const std::exception& e) {
        std::cerr << "C++ error in scipy_solver: " << e.what() << std::endl;
        throw;
    }
}

void save_checkpoint(const std::string& filename,
                    const Eigen::VectorXcd& params,
                    const Eigen::MatrixXd& subspace,
                    const std::string& args) {
    py::gil_scoped_acquire acquire;

    try {
        py::module np = py::module::import("numpy");

        py::object params_np = py::cast(params);
        py::object subspace_np = py::cast(subspace);

        py::dict save_dict;
        save_dict["params"] = params_np;
        save_dict["subspace"] = subspace_np;
        save_dict["args"] = args;

        np.attr("savez_compressed")(filename, **save_dict);
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error during save:\n" << e.what() << std::endl;
        throw std::runtime_error("Failed to save checkpoint");
    }
}

std::string load_checkpoint(const std::string& filename, Eigen::VectorXcd &params, Eigen::MatrixXd &subspace_vec) {
    py::gil_scoped_acquire acquire;

    try {
        py::module np = py::module::import("numpy");

        py::object npz_file = np.attr("load")(filename);

        params = npz_file.attr("get")("params").cast<Eigen::VectorXcd>();

        subspace_vec = npz_file.attr("get")("subspace").cast<Eigen::MatrixXd>();

        return std::string("");
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error during load:\n" << e.what() << std::endl;
        throw std::runtime_error("Failed to load checkpoint");
    }
}

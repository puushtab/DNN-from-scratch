#include "../include/losses.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

double LossMSE::forward(MatrixXd inputs, MatrixXd targets){
    assert(inputs.rows() == targets.rows() && inputs.cols() == targets.cols() && "Inputs and targets must have the same dimensions");
    this->inputs = inputs;
    this->targets = targets;

    auto square = [](double x) { return x * x; }; // Function to apply
    MatrixXd squared_diff = (inputs - targets).unaryExpr(square);
    double mean_squared_error = squared_diff.mean(); // Mean sur les colonnes (inputs) et lignes (samples)
    return mean_squared_error;
}

// derivative of MSE w.r.t inputs: 2*(inputs - targets)/N
MatrixXd LossMSE::backward(){
    int N = inputs.rows();
    int batch_size = inputs.cols();
    return (2.0 / (N * batch_size)) * (inputs - targets);
}

// TODO: Cross-entropy loss
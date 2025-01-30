#include "../include/activations.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

MatrixXd ActivationReLU::forward(const MatrixXd& inputs){
    this->inputs = inputs;
    batch_size = inputs.cols();
    
    return inputs.cwiseMax(0);
}

// Backward: derivative of ReLU(x) is 1 if x>0 else 0
// grad shape: (output_size)
MatrixXd ActivationReLU::backward(const MatrixXd& grad){
    return (inputs.array() > 0).cast<double>().array() * grad.array();
}

MatrixXd ActivationSigmoid::forward(const MatrixXd& inputs){
    this->inputs = inputs;
    batch_size = inputs.cols();

    auto func = [](double x) { return 1 / (1 + exp(-x)); }; // Correct sigmoid function
    return inputs.unaryExpr(func); // Apply function to each element of the matrix
}

MatrixXd ActivationSigmoid::backward(const MatrixXd& grad){
    // We need the sigmoid of inputs again
    MatrixXd s = forward(inputs);
    MatrixXd ds = s.array() * (1 - s.array()); // derivative of sigmoid
    // Multiply elementwise by incoming gradient
    return ds.array() * grad.array();
}


/* TODO */
MatrixXd ActivationSoftmax::forward(const MatrixXd& inputs){
    return inputs.array().exp() / inputs.array().exp().sum();
}
MatrixXd ActivationSoftmax::backward(const MatrixXd& grad){
    /* NOT IMPLEMENTED YET !!!!*/
    return grad;
}
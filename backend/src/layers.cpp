#include "../include/layers.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

/* Class of a DenseLayer*/  
MatrixXd DenseLayer::forward(MatrixXd inputs){
    this->inputs = inputs;
    batch_size = inputs.cols();
    
    MatrixXd bias_matrix = bias * MatrixXd::Ones(1, batch_size);
    // std::cout << "weights_shape: (" << weights.rows() << ", " << weights.cols() << ")" << std::endl;
    // std::cout << "inputs_shape: (" << inputs.rows() << ", " << inputs.cols() << ")" << std::endl;
    // std::cout << "bias_shape: (" << bias_matrix.rows() << ", " << bias_matrix.cols() << ")" << std::endl;
    return weights * inputs + bias_matrix; /* Attention transposée !*/
}

// Backward pass
// grad shape: (output_size), i.e. dLoss/dOutput of this layer
// returns shape: (input_size), i.e. dLoss/dInput for the previous layer
MatrixXd DenseLayer::backward(const MatrixXd& grad){
    // 1) dBias = gradient from next layer (for each neuron)
    //   We sum the gradients for each sample in the batch WITHOUT DOING MEAN
    dbiases = grad * MatrixXd::Ones(batch_size, 1);

    // 2) dWeights = grad (output_size,1) * inputs^T (1,input_size) => (output_size,input_size)
    //    This is the outer product WITHOUT A MEAN
    dweights = grad * inputs.transpose();

    // 3) dInputs = W^T (input_size,output_size) * grad (output_size) => (input_size, batch_size) WITHOUT A MEAN
    dinputs = weights.transpose() * grad;

    return dinputs;
}

// Simple gradient-descent parameter update
void DenseLayer::updateParameters(double learning_rate){
    weights -= learning_rate * dweights;
    bias -= learning_rate * dbiases;
}
    

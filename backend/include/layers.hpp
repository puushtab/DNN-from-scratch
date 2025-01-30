#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <Eigen/Dense>
#include <random>

using namespace Eigen;

class Layer{
    public:
        int input_size; // Size of the input
        int output_size; // Number of neurons
        MatrixXd weights;
        VectorXd bias; // Store the bias for each neuron

        MatrixXd inputs; // Store the inputs for backpropagation, matrix of shape (input_size, batch_size)
        int batch_size; // Number of samples in a batch
        MatrixXd dweights;
        VectorXd dbiases;
        MatrixXd dinputs;

    public:
        Layer(): input_size(0), output_size(0), weights(MatrixXd()), bias(VectorXd()) {}
        Layer(int input_size, int output_size, MatrixXd weights, VectorXd bias): input_size(input_size), output_size(output_size), 
                                                                                      weights(weights), bias(bias) {}
        Layer(int input_size, int output_size): input_size(input_size), output_size(output_size), 
                                                     bias(VectorXd::Zero(output_size)), weights(MatrixXd::Random(output_size, input_size)) {}
        Layer(const Layer& other) 
            : input_size(other.input_size), output_size(other.output_size), 
              weights(other.weights), bias(other.bias), 
              inputs(other.inputs), batch_size(other.batch_size), 
              dweights(other.dweights), dbiases(other.dbiases), dinputs(other.dinputs) {}
        virtual MatrixXd forward(MatrixXd inputs) = 0;
        virtual MatrixXd backward(const MatrixXd& grad) = 0;
        virtual void updateParameters(double learning_rate) = 0;
};

class DenseLayer: public Layer {
    public:
        DenseLayer(): Layer() {}
        DenseLayer(int input_size, int output_size): Layer(input_size, output_size) {}
        DenseLayer(int input_size, int output_size, MatrixXd weights, VectorXd bias): Layer(input_size, output_size, weights, bias) {}
        MatrixXd forward(MatrixXd inputs) override;
        // Backward pass
        // grad shape: (output_size), i.e. dLoss/dOutput of this layer
        // returns shape: (input_size), i.e. dLoss/dInput for the previous layer
        MatrixXd backward(const MatrixXd& grad) override;
        // Simple gradient-descent parameter update
        void updateParameters(double learning_rate) override;
};  
#endif 
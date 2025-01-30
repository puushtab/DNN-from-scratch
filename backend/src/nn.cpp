#include "../include/nn.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

using namespace Eigen;

MatrixXd NeuralNetwork::forward(MatrixXd features){
    // ---------------------------
    // Forward pass
    // ---------------------------
    MatrixXd x = features;
    for (int k = 0; k < n_layers; k++){
        MatrixXd z = layers[k]->forward(x);
        MatrixXd a = activations[k]->forward(z);
        x = a;
    }
    return x;
}

void NeuralNetwork::backward(MatrixXd a, MatrixXd labels){
    // Compute loss
    double loss_value = loss.forward(a, labels);
    // ---------------------------
    // Backward pass
    // ---------------------------
    // 1) Loss derivative w.r.t the final activation output
    MatrixXd dLoss_da = loss.backward();
    for (int k = n_layers - 1; k >= 0; k--){
        MatrixXd dLoss_dz = activations[k]->backward(dLoss_da);
        dLoss_da = layers[k]->backward(dLoss_dz);
    }
}
void NeuralNetwork::updateParameters(double learning_rate){
    // ---------------------------
    // Gradient descent update
    // ---------------------------
    for(int k = 0; k < n_layers; k++){
        layers[k]->updateParameters(learning_rate);
    }
}

void NeuralNetwork::updatePredMatrix(double range, int n_discretization){
    MatrixXd grid_points(2, n_discretization*n_discretization);
    double step = 2*range/n_discretization;

    int index = 0;
    for (int i = 0; i < n_discretization; ++i) {
        for (int j = 0; j < n_discretization; ++j) {
            if(index < n_discretization*n_discretization){
                double x = -range + j * step;
                double y = -range + i * step;
                grid_points(0, index) = x;
                grid_points(1, index) = y;
                index++;
            }
        }
    }
    // std::cout << "grid:" << grid_points << std::endl;
    this->prediction_matrix = forward(grid_points);
}

void NeuralNetwork::train(MatrixXd features, MatrixXd labels, int epochs, double learning_rate, int n_batch, int batch_size){
    train_features = features;
    train_labels = labels;
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffleMatrixColumns(train_features, train_labels); //Stochastic aspect of the optimizer
        for(int j = 0; j < n_batch; j++){
            // batch_size points selected randomly
            int start_index = j * batch_size;
            int end_index = std::min(start_index + batch_size, (int)train_features.cols());
            MatrixXd x = train_features.block(0, start_index, input_size, end_index - start_index);
            MatrixXd y = train_labels.block(0, start_index, output_size, end_index - start_index);
            
            // Assert to check dimensions
            assert(x.rows() == input_size && x.cols() == batch_size && "x dimensions must match input_size and batch_size");
            assert(y.rows() == output_size && y.cols() == batch_size && "y dimensions must match output_size and batch_size");

            // Forward pass
            MatrixXd predictions = forward(train_features);
            // Backward pass
            backward(predictions, train_labels);
            // Update parameters
            updateParameters(learning_rate);
            // Optionally, print the loss for monitoring. Computed twice
            double loss_value = loss.forward(predictions, train_labels);
            // std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << loss_value << std::endl;
        }
    }
}

MatrixXd NeuralNetwork::predict(MatrixXd x){
    // Forward pass
    MatrixXd predictions = forward(x);
    return predictions;
}

double NeuralNetwork::evaluate(MatrixXd test_features, MatrixXd test_labels, int n_test_samples){
    // Forward pass
    MatrixXd predictions_double = forward(test_features);

    // Compute loss
    double loss_value = loss.forward(predictions_double, test_labels);

    MatrixXd predictions = predictions_double.unaryExpr(&binaryClassification);

    // Testing, we calculate the accuracy of the model
    double accuracy = 0.0;
    for(int i = 0; i < n_test_samples; i++){
        // std::cout << "Prediction: " << predictions(0, i) << ", Label: " << test_labels(0, i) << std::endl;
        if(predictions(0, i) == test_labels(0, i)){
            accuracy += 1.0;
        }
    }
    accuracy /= n_test_samples;
    std::cout << "Test accuracy: " << accuracy << std::endl;
    std::cout << "Test loss: " << loss_value << std::endl;

    return accuracy;
}
void NeuralNetwork::printDetails() const {
    std::cout << "Neural Network Details:" << std::endl;
    std::cout << "Input Size: " << input_size << std::endl;
    std::cout << "Output Size: " << output_size << std::endl;
    std::cout << "Number of Layers: " << n_layers << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;

    for (int i = 0; i < n_layers; ++i) {
        std::cout << "Layer " << i << ":" << std::endl;
        std::cout << "Weights:\n" << layers[i]->weights << std::endl;
        std::cout << "Biases:\n" << layers[i]->bias << std::endl;
    }

    std::cout << "Prediction Matrix:\n" << prediction_matrix << std::endl;
}
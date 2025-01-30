#include "../include/layers.hpp"
#include "../include/activations.hpp"
#include "../include/losses.hpp"
#include "../include/utils.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

class NeuralNetwork{
    public:

        double accuracy;
        int input_size;
        int output_size;

        int n_layers;
        vector<unique_ptr<Layer>> layers;
        vector<unique_ptr<Activation>> activations;
        LossMSE loss; //TODO : Make it generic
        double (*classification)(double);
        
        int epochs;
        int batch_size;
        int n_batch;

        // MatrixXd* grads;
        // MatrixXd* dweights;
        // VectorXd* dbiases;
        // MatrixXd* dinputs;
        double learning_rate;

        MatrixXd train_features;
        MatrixXd train_labels;
        MatrixXd test_features;
        MatrixXd test_labels;

        MatrixXd prediction_matrix;

    public:
        NeuralNetwork() = default;
        NeuralNetwork(int input_size, int output_size, int* hidden_size, int n_layers, int epochs, double learning_rate, double (*classification)(double)){
            //TODO: Take layers as input
            
            this->input_size = input_size;
            this->output_size = output_size;
            this->n_layers = n_layers;
            this->epochs = epochs;
            this->learning_rate = learning_rate;
            this->classification = classification;

            layers = vector<unique_ptr<Layer>>(n_layers);
            activations = vector<unique_ptr<Activation>>(n_layers);
            // grads = new MatrixXd[n_layers];
            // dweights = new MatrixXd[n_layers];
            // dbiases = new VectorXd[n_layers];
            // dinputs = new MatrixXd[n_layers];

            // Initialize layers
            layers[0] = make_unique<DenseLayer>(input_size, hidden_size[0]);
            activations[0] = make_unique<ActivationReLU>();
            for(int i = 1; i < n_layers - 1; i++){
                layers[i] = make_unique<DenseLayer>(hidden_size[i-1], hidden_size[i]);
                activations[i] = make_unique<ActivationReLU>();
            }
            layers[n_layers - 1] = make_unique<DenseLayer>(hidden_size[n_layers - 2], output_size);
            activations[n_layers - 1] = make_unique<ActivationSigmoid>();

            prediction_matrix = MatrixXd::Zero(100, 100);
        }
        NeuralNetwork(int input_size, int output_size, int* hidden_size, int n_layers, int epochs, double learning_rate, double (*classification)(double),  std::string activation){
            //TODO: Take layers as input
            
            this->input_size = input_size;
            this->output_size = output_size;
            this->n_layers = n_layers;
            this->epochs = epochs;
            this->learning_rate = learning_rate;
            this->classification = classification;

            layers = vector<unique_ptr<Layer>>(n_layers);
            activations = vector<unique_ptr<Activation>>(n_layers);
            // grads = new MatrixXd[n_layers];
            // dweights = new MatrixXd[n_layers];
            // dbiases = new VectorXd[n_layers];
            // dinputs = new MatrixXd[n_layers];

            // Initialize layers
            if(activation == "ReLU"){
                activations[0] = make_unique<ActivationReLU>();
                for(int i = 1; i < n_layers - 1; i++){
                    activations[i] = make_unique<ActivationReLU>();
                }
                activations[n_layers - 1] = make_unique<ActivationSigmoid>();
            }
            else if(activation == "Sigmoid"){
                activations[0] = make_unique<ActivationSigmoid>();
                for(int i = 1; i < n_layers - 1; i++){
                    activations[i] = make_unique<ActivationSigmoid>();
                }
                activations[n_layers - 1] = make_unique<ActivationSigmoid>();
            }
            else{
                cout << "Unknown activation function" << endl;
            }
            
            layers[0] = make_unique<DenseLayer>(input_size, hidden_size[0]);
            for(int i = 1; i < n_layers - 1; i++){
                layers[i] = make_unique<DenseLayer>(hidden_size[i-1], hidden_size[i]);
            }
            layers[n_layers - 1] = make_unique<DenseLayer>(hidden_size[n_layers - 2], output_size);

            prediction_matrix = MatrixXd::Zero(100, 100);
        }
        ~NeuralNetwork(){
            // No need to manually delete unique_ptr members as they will be automatically cleaned up
        }
        NeuralNetwork(const NeuralNetwork& other)
            : input_size(other.input_size), output_size(other.output_size),
              n_layers(other.n_layers), epochs(other.epochs), 
              learning_rate(other.learning_rate), classification(other.classification),
              train_features(other.train_features), train_labels(other.train_labels),
              test_features(other.test_features), test_labels(other.test_labels),
              prediction_matrix(other.prediction_matrix) {
            
            layers = vector<unique_ptr<Layer>>(n_layers);
            activations = vector<unique_ptr<Activation>>(n_layers);
            
            for (int i = 0; i < n_layers; ++i) {
                layers[i] = make_unique<DenseLayer>(*dynamic_cast<DenseLayer*>(other.layers[i].get()));
                if (dynamic_cast<ActivationReLU*>(other.activations[i].get())) {
                    activations[i] = make_unique<ActivationReLU>(*dynamic_cast<ActivationReLU*>(other.activations[i].get()));
                } else if (dynamic_cast<ActivationSigmoid*>(other.activations[i].get())) {
                    activations[i] = make_unique<ActivationSigmoid>(*dynamic_cast<ActivationSigmoid*>(other.activations[i].get()));
                }
            }
        }

        MatrixXd forward(MatrixXd features);
        void backward(MatrixXd a, MatrixXd labels);
        void updatePredMatrix(double range, int n_discretization);
        void updateParameters(double learning_rate);
        void train(MatrixXd features, MatrixXd labels, int epochs, double learning_rate, int n_batch, int batch_size);
        MatrixXd predict(MatrixXd x);
        double evaluate(MatrixXd test_features, MatrixXd test_labels, int n_test_samples);
        double getLoss() {
            return this->loss.forward(test_features, test_labels);
        }
        double getLoss(MatrixXd predictions, MatrixXd labels) {
            return this->loss.forward(predictions, labels);
        }
        MatrixXd getPredictionMatrix() {
            return this->prediction_matrix;
        }
        void printDetails() const;
        double range(MatrixXd features){
            return features.maxCoeff() - features.minCoeff();
        }
        double range(){
            return train_features.maxCoeff() - train_features.minCoeff();
        }
};
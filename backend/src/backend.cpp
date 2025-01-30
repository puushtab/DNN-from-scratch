#include "../include/backend.hpp"

using namespace Eigen;

NeuralNetwork training(int nbLayers, int* nbNeurons, std::string lossfunction, std::string activation, double learning_rate, double training_ratio, int batch_size, int epochs, int epoch_push, std::string optimizer, std::string dataset, ThreadSafeQueue<NeuralNetwork>& networkQueue){
    // Only 1 LossFunction for the moment
    // 2 possibles activation functions
    // LearningRate ok
    // batch_size ignored atm
    // 1 optimizer possible
    // 4 dataset

    // Choose dataset
    std::string filename;
    if (dataset == "Circles") {
        filename = "data/dataset_circles.csv";
    } else if (dataset == "Blobs") {
        filename = "data/dataset_blobs_close.csv";
    } else if (dataset == "Moons") {
        filename = "data/dataset_moons.csv";
    } else if (dataset == "Linear") {
        filename = "data/dataset_linear.csv";
    } else {
        std::cerr << "Unknown dataset: " << dataset << std::endl;
        std::__throw_invalid_argument;
    }
    std::cout << filename << std::endl;
    std::vector<std::vector<double>> features0;
    std::vector<std::vector<double>> labels0;
    readCSV(filename, features0, labels0); // Read data from CSV file
    
    int n_samples = features0.size();
    int input_size = features0[0].size();
    int output_size = labels0[0].size();

    MatrixXd features = vectorToMatrix(features0).transpose();
    MatrixXd labels = vectorToMatrix(labels0).transpose();

    std::cout << "Features" << std::endl << features << std::endl;
    std::cout << "Labels" << std::endl << labels << std::endl;
    
    int n_training_samples = (int)(n_samples * training_ratio);
    int n_testing_samples = n_samples - n_training_samples;

    MatrixXd training_features = features.block(0, 0, input_size, n_training_samples);
    MatrixXd training_labels = labels.block(0, 0, output_size, n_training_samples);
    MatrixXd testing_features = features.block(0, n_training_samples, input_size, n_testing_samples);
    MatrixXd testing_labels = labels.block(0, n_training_samples, output_size, n_testing_samples);
    
    if(batch_size == 700){
        batch_size = n_training_samples;
    }
    int n_batch = (int)(n_training_samples / batch_size); //ATTENTION, a part of the data is lost !
    double epsilon_gradient = 1e-8;
    double gradient_norm = 1e8;

    std::cout << "n_samples: " << n_samples << std::endl;
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "epochs: " << epochs << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "learning_rate: " << learning_rate << std::endl;
    std::cout << "training_ratio: " << training_ratio << std::endl; 
    std::cout << "x_shape: (" << features.rows() << ", " << features.cols() << ")" << std::endl;
    std::cout << "y_shape: (" << labels.rows() << ", " << labels.cols() << ")" << std::endl;

    // For the prediction matrix
    double range = (training_features.maxCoeff() - training_features.minCoeff());

    int n_discretization = 100;
    std::cout << "range: " << range << std::endl;

    // Define the hidden layers sizes
    input_size = input_size;
    output_size = nbNeurons[nbLayers-1];

    int* hidden_sizes = new int[nbLayers - 1];
    for (int i = 0; i < nbLayers - 1; ++i) {
        hidden_sizes[i] = nbNeurons[i];
    }
    
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "output_size: " << output_size << std::endl;
    std::cout << "hidden_sizes: ";
    for (int i = 0; i < nbLayers - 1; ++i) {
        std::cout << hidden_sizes[i] << " ";
    }
    std::cout << std::endl;
    
    // Create the neural network
    NeuralNetwork nn(input_size, output_size, hidden_sizes, nbLayers, epochs, 
                        learning_rate, binaryClassification, activation);
    nn.train_features = training_features;
    nn.train_labels = training_labels;
    nn.test_features = testing_features;
    nn.test_labels = testing_labels;

    for(int epoch=0; epoch < epochs; epoch++){
        // Train the neural network
        nn.train(training_features, training_labels, 1, learning_rate, n_batch, batch_size);
        if (epoch%epoch_push == 0){
            std::cout << "Epoch push: " << epoch << " - Loss: " << nn.loss.forward(nn.forward(training_features), training_labels) << std::endl;
            networkQueue.push(nn);
        }
    }
    nn.updatePredMatrix(range, n_discretization);

    // Evaluate the neural network
    double accuracy = nn.evaluate(testing_features, testing_labels, n_testing_samples);

    std::cout << "Accuracy: " << accuracy << std::endl;

    nn.accuracy = accuracy;
    
    delete hidden_sizes;

    return nn;
}

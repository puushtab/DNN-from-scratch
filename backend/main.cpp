#include "include/backend.hpp"

int main() {
    int nb_layers = 3;
    int nb_neurons[3] = {4,4,1};
    string loss_function = "MSE";
    string activationfunc = "ReLU";
    double learning_rate = 0.1;
    double training_rate = 0.7;
    double batch_size = 100;
    double epochs = 10000;
    int epoch_push = 100;
    string optimizer = "SGD";
    string dataset = "Blobs";
    ThreadSafeQueue< NeuralNetwork > networkQueue; 
    
    training(nb_layers, nb_neurons, loss_function, activationfunc, learning_rate, training_rate, batch_size, epochs, epoch_push, "SGD", dataset, networkQueue);
    return 0;
}
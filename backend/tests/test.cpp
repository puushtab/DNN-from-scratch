#include <iostream>
#include "../include/nn.hpp"  // Include your NeuralNetwork header

int main() {
    // Example hidden layer sizes
    int hidden_sizes[] = {4, 2};
    // Create a NeuralNetwork instance (2 layers, for example)
    NeuralNetwork net1(3, 1, hidden_sizes, 3, 10, 0.01, nullptr);

    // Print initial weights of net1 (layer 0)
    std::cout << "net1.layers[0]->weights before copy:\n" 
              << net1.layers[0]->weights << std::endl;

    // Make a deep copy using the copy constructor
    NeuralNetwork net2 = net1;

    // Modify net2's first layer weight
    net2.layers[0]->weights(0, 0) = 9999.99;

    // Print weights again to verify that net1 is unchanged
    std::cout << "net1.layers[0]->weights after modifying net2:\n" 
              << net1.layers[0]->weights << std::endl;
    std::cout << "net2.layers[0]->weights:\n" 
              << net2.layers[0]->weights << std::endl;

    return 0;
}
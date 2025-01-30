#include "layers.hpp"
#include "losses.hpp"
#include "activations.hpp"
#include "utils.hpp"
#include "nn.hpp"
#include "../../frontend/include/queue.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include <random>
#include <chrono>
#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <unistd.h>
#include <math.h>
#include <memory>
#include <thread>

NeuralNetwork training(int nbLayers, int* nbNeurons, std::string lossfunction, std::string activation, double learning_rate, double training_ratio, int batch_size, int epochs, int epoch_push, std::string optimizer, std::string dataset, ThreadSafeQueue<NeuralNetwork>& networkQueue);
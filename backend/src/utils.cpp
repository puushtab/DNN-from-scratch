#include "../include/utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string> 
#include <Eigen/Core>
#include <Eigen/Sparse>

MatrixXd createRandomPositiveMatrix(size_t rows, size_t cols) {
    // Create a random matrix with values in the range [-1, 1]
    MatrixXd matrix = MatrixXd::Random(rows, cols);
    // Transform values from [-1, 1] to [0, 1]
    matrix = (matrix + MatrixXd::Ones(rows, cols)) / 2.0;
    return matrix;
}

VectorXd stdVectorToEigenVectorXd(const std::vector<double>& vec) {
    return Map<const VectorXd>(vec.data(), vec.size());
}

MatrixXd vectorToMatrix(const std::vector<std::vector<double>>& vec) {
    if (vec.empty()) {
        return MatrixXd(0, 0);
    }
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    MatrixXd matrix(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix(i, j) = vec[i][j];
        }
    }
    return matrix;
}
// Function to read CSV file and store it in vectors
void readCSV(const std::string& filename, std::vector<std::vector<double>>& features, std::vector<std::vector<double>>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return;
    }

    std::string line;
    bool headerSkipped = false;
    
    while (std::getline(file, line)) {
        // Skip header
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        std::stringstream ss(line);
        std::vector<double> features_row;
        std::vector<double> labels_row;
        std::string value;
        int colIndex = 0;
        
        while (std::getline(ss, value, ',')) {
            if (colIndex < 2) {
                features_row.push_back(std::stod(value));  // Convert to float for features
            } else {
                labels_row.push_back(std::stod(value));  // Convert to float for labels
            }
            colIndex++;
        }
        features.push_back(features_row);
        labels.push_back(labels_row);
    }
    file.close();
}

double binaryClassification(double a){
    if(a > 0.5){
        return 1;
    }
    return 0;
}

void shuffleMatrixColumns(MatrixXd& matrix) {
    MatrixXd shuffledMatrix = matrix;
    std::vector<int> indices(matrix.cols());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    for (size_t i = 0; i < indices.size(); ++i) {
        shuffledMatrix.col(i) = matrix.col(indices[i]);
    }
    matrix = shuffledMatrix;
}

void shuffleMatrixColumns(MatrixXd& features, MatrixXd& labels) {
    // Ensure both matrices have the same number of columns
    assert(features.cols() == labels.cols());

    // Create a vector of column indices
    std::vector<int> indices(features.cols());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices
    std::random_shuffle(indices.begin(), indices.end());

    // Apply the shuffled indices to both features and labels
    MatrixXd shuffledFeatures(features.rows(), features.cols());
    MatrixXd shuffledLabels(labels.rows(), labels.cols());

    for (size_t i = 0; i < indices.size(); ++i) {
        shuffledFeatures.col(i) = features.col(indices[i]);
        shuffledLabels.col(i) = labels.col(indices[i]);
    }

    // Update the input matrices
    features = shuffledFeatures;
    labels = shuffledLabels;
}
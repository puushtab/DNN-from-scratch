#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>

using namespace Eigen;

double sigmoid(double x);
MatrixXd createRandomPositiveMatrix(size_t rows, size_t cols);
VectorXd stdVectorToEigenVectorXd(const std::vector<double>& vec);
MatrixXd vectorToMatrix(const std::vector<std::vector<double>>& vec);
void readCSV(const std::string& filename, std::vector<std::vector<double>>& features, std::vector<std::vector<double>>& labels);
double binaryClassification(double a);
void shuffleMatrixColumns(MatrixXd& matrix);
void shuffleMatrixColumns(MatrixXd& features, MatrixXd& labels);

#endif // UTILS_H
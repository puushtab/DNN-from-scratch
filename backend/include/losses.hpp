#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <Eigen/Dense>
#include <random>

using namespace Eigen;

class Loss{
    public:
        MatrixXd inputs;
        MatrixXd targets;
        int batch_size;
    public:
        Loss(){}
        virtual double forward(MatrixXd inputs, MatrixXd targets) = 0;
        virtual MatrixXd backward() = 0;
};

class LossMSE: public Loss{ 
    public:
        double forward(MatrixXd inputs, MatrixXd targets) override;
        MatrixXd backward() override;
};
#endif
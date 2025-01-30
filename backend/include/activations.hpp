#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <Eigen/Dense>

#include <random>

using namespace Eigen;

class Activation{
    public:
        MatrixXd inputs;
        int batch_size;
    public:
        Activation() : batch_size(1) {}
        virtual MatrixXd forward(const MatrixXd& inputs) = 0;
        virtual MatrixXd backward(const MatrixXd& grad) = 0;
};

class ActivationReLU: public Activation{
    public:
        ActivationReLU() : Activation() {}
        MatrixXd forward(const MatrixXd& inputs) override;

        // Backward: derivative of ReLU(x) is 1 if x>0 else 0
        // grad shape: (output_size)
        MatrixXd backward(const MatrixXd& grad) override;
};

class ActivationSigmoid: public Activation{
    public:
        ActivationSigmoid() : Activation() {}
        MatrixXd forward(const MatrixXd& inputs) override;
        MatrixXd backward(const MatrixXd& grad) override;
};

/* TODO */
class ActivationSoftmax: public Activation{
    public:
        ActivationSoftmax() : Activation() {}
        MatrixXd forward(const MatrixXd& inputs) override;
        MatrixXd backward(const MatrixXd& grad) override;
};
#endif
#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_
#include <vector>
#include "activation.hpp"
#include "matrix/matrix.hpp"
using namespace std;

class NeuralNetwork {
    vector<Matrix> weights_;
    vector<Matrix> h_;
    double training_rate_;
    f f_;
    f df_;

    public:
        NeuralNetwork(vector<Matrix> weights, ActivationEnum act, double training_rate);
        Matrix forward(const Matrix& inputs);
        void backward(Matrix outputs);
        Matrix getOutput() const;
        vector<Matrix> getWeights() const;
};

#endif

#include "neuralnetwork.hpp"
#include "activation.hpp"
#include "matrix/matrix.hpp"
using namespace std;

NeuralNetwork::NeuralNetwork(vector<Matrix> weights, ActivationEnum act, double training_rate): weights_(weights) {
    training_rate_ = training_rate;

    // Set activation functions
    f_ = getF(act);
    df_ = getdF(act);

    // Initialize internal state matrices
    for (vector<Matrix>::size_type i = 0; i < weights_.size(); i++) {
        h_.push_back(Matrix(1, weights_[i].getDimX()));
    }
    h_.push_back(Matrix(1, weights_.back().getDimY()));
}

vector<Matrix> NeuralNetwork::getWeights() const {
    return weights_;    
}

Matrix NeuralNetwork::getOutput() const {
    return h_.back();
}

Matrix NeuralNetwork::forward(const Matrix& inputs) {
    // Store input
    h_[0] = inputs;

    // Propogate through hidden layers
    for (vector<Matrix>::size_type i = 0; i < weights_.size(); i++) {
        Matrix layer_input = h_[i] * weights_[i];
        h_[i+1] = layer_input.f(f_);
    }

    return getOutput();
}

void NeuralNetwork::backward(Matrix outputs) {
    // Calculate activation derivatives
    // dNout/dNin = f'(in)
    vector<Matrix> dNout_dNin;
    for (vector<Matrix>::size_type i = 0; i < weights_.size(); i++) { 
        Matrix layer_input = h_[i] * weights_[i];
        dNout_dNin.push_back(layer_input.f(df_));
    }

    // Backpropagate error derivatives wrt node output
    // dE/dNout[i] = (dE/dNout[i+1])(dNout/dNin[i+1])(dNin_dNout[i+1])
    //   = (dE/dNout[i+1])(f'[i+1])(W[i+1])
    vector<Matrix> dE_dNout;
    dE_dNout.push_back(h_.back() - outputs);
    for (vector<Matrix>::size_type i = weights_.size() - 2; i < weights_.size() - 1; i--) {
        dE_dNout.insert(dE_dNout.begin(), 
            dE_dNout.front().elementMultiplies(dNout_dNin[i+1]) * weights_[i+1].transpose());
    }

    // Calculate error derivatives wrt weight and update weights
    // dE/dW = n(dE_dNout)(dNout_dNin)(dNin_dW)
    //   =n(dE_dNout)(f')(h)
    for (vector<Matrix>::size_type i = 0; i < weights_.size(); i++) {
        Matrix dE_dW = 
            (dE_dNout[i].elementMultiplies(dNout_dNin[i])).transpose() * h_[i];
        Matrix delta_W = dE_dW * training_rate_;
        weights_[i] = weights_[i] - delta_W.transpose();
    }
}


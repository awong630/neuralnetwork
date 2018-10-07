#include <iostream>
#include <random>
#include <cmath>
#include "neuralnetwork.hpp"
#include "activation.hpp"
#include "matrix/matrix.hpp"

double normalize(double value, double bias, double factor) {
    return (value + bias) / factor;
}

double denormalize(double value, double bias, double factor) {
    return value * factor - bias;
}

int main() {
    // Parameters
    int training_cases = 1000000;
    int training_seed = 123;
    double training_min = 0;
    double training_max = 100;
    double x_bias = 0;
    double x_factor = 100;
    double y_bias = 0;
    double y_factor = 10;

    // Setup neural network
    Matrix layer1 = Matrix::random( 1, 25, -1, 1, 1);
    Matrix layer2 = Matrix::random(25, 25, -1, 1, 2);
    Matrix layer3 = Matrix::random(25,  1, -1, 1, 3);
    vector<Matrix> weights {layer1, layer2, layer3};
    NeuralNetwork net(weights, sigmoid, 0.5);

    // Run training
    std::mt19937 mt(training_seed);
    std::uniform_real_distribution<double> dist(training_min, 
        std::nextafter(training_max, std::numeric_limits<double>::max()));
    vector<double> x;
    vector<double> y;
    for (int i = 0; i < training_cases; i++) {
        double x = dist(mt);
        double y = sqrt(x);
        Matrix in(1, 1, normalize(x, x_bias, x_factor));
        net.forward(in);
        Matrix out(1, 1, normalize(y, y_bias, y_factor));
        net.backward(out);
    }

    // Output test calculations
    vector<double> test_data {100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0};
    for (std::vector<double>::iterator it = test_data.begin(); it != test_data.end(); ++it) {
        double x = *it;
        Matrix in(1, 1, normalize(x, x_bias, x_factor));
        net.forward(in);
        Matrix out = net.getOutput();
        double y = denormalize(out(0, 0), y_bias, y_factor);
        std::cout << "Sqrt of " << x << ": " << y << endl;
    }
}


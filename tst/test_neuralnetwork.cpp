#include <iostream>
#include "neuralnetwork.hpp"
#include "activation.hpp"
#include "catch/catch.hpp"
#include "matrix/matrix.hpp"
using namespace std;

TEST_CASE("Forward and Backward") {
    vector<double> weights1 {1, 2};
    vector<double> weights2 {3, 1};
    Matrix layer1(1, 2, weights1);
    Matrix layer2(2, 1, weights2);
    vector<Matrix> weights {layer1, layer2};
    NeuralNetwork net(weights, relu, 0.01);
    vector<double> input_array {1};
    Matrix input(1, 1, input_array);
    Matrix outputs = net.forward(input);
    REQUIRE(outputs(0, 0) == 5);

    vector<double> training {4};
    Matrix training_output(1, 1, training);
    net.backward(training_output);
    vector<Matrix> new_weights = net.getWeights();
    REQUIRE(new_weights[0](0, 0) == 0.97);
    REQUIRE(new_weights[0](0, 1) == 1.99);
    REQUIRE(new_weights[1](0, 0) == 2.99);
    REQUIRE(new_weights[1](1, 0) == 0.98);
}

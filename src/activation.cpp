#include "activation.hpp"
#include <algorithm>
#include <cmath>
using namespace std;

f getF(ActivationEnum act) {
    switch (act) {
        case relu:    return activation::relu;
        case sigmoid: return activation::sigmoid;
    }
}

f getdF(ActivationEnum act) {
    switch (act) {
        case relu:    return activation::d_relu;
        case sigmoid: return activation::d_sigmoid;
    }
}

double activation::relu(double x) {
    return max(0.0, x);
}

double activation::d_relu(double x) {
    if (x > 0) {
        return 1;
    }
    else {
        return 0;
    }
}

double activation::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double activation::d_sigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

#ifndef ACTIVATION_HPP_
#define ACTIVATION_HPP_

enum ActivationEnum {relu, sigmoid};

typedef double (*f)(double);

f getF(ActivationEnum act);
f getdF(ActivationEnum act);

namespace activation {
    double relu(double x);
    double d_relu(double x);
    double sigmoid(double x);
    double d_sigmoid(double x);
}

#endif

#include"matrix.h"
#include<iostream>
#include<functional>
#include<cmath>

using vec = pp::vector;
using mat = pp::matrix;

int current_iter = 0; // Global variable to count iterations

vec gradient(std::function<double(const vec&)> f, vec& x_input){
    vec x1 = x_input; // make a copy of the input vector
    vec x2 = x_input; // make a copy of the input vector
    vec gf(x1.size());
    for (int i =0; i < x1.size(); ++i) { // compute the i-th component of the gradient
        double dxi = (1+fabs(x1[i]))*(pow(2, -13));
        x1[i] += dxi;
        x2[i] -= dxi; // perturb x1[i] and x2[i] 
        gf[i] = (f(x1) - f(x2)) / (2 * dxi); // compute the i-th component of the gradient (Should be more accurate than using a single perturbation)
        x1[i] -= dxi; // restore x[i]
        x2[i] += dxi; // restore x2[i]
    }
    return gf; //return the gradient vector
}

mat hessian(std::function<double(const vec&)> f, vec& x_input){
    vec x1 = x_input; // make a copy of the input vector
    vec x2 = x_input; // make a copy of the input vector
    int n = x1.size();
    mat H(n, n);
    vec gfx = gradient(f,x1);
    for (int j = 0; j < n; ++j) {
        double dxj = (1+fabs(x1[j]))*(pow(2, -13));
        x1[j] += dxj; // perturb x1[j]
        x2[j] -= dxj; // perturb x2[j]
        vec dgf = gradient(f,x1) - gradient(f, x2); // compute the difference in gradients
        for (int i = 0; i < n; ++i) {
            H(j, i) = dgf[i] / (2 * dxj); // compute the i-th row and j-th column of the hessian (Should be more accurate than using a single perturbation)
        }
        x1[j] -= dxj; // restore x1[j]
        x2[j] += dxj; // restore x2[j]
    }
    return H; //return the hessian matrix
}

vec newton(std::function<double(const vec&)> f, vec& x_input, double acc, int max_iter) {
    vec x = x_input; // make a copy of the input vector
    vec delta_x;
    while (current_iter < max_iter) {
        vec g = gradient(f, x);
        if (g.norm() < acc) {
            return x; // return the current point if the gradient is small enough
        }
        mat H = hessian(f, x);
        try {
            delta_x = pp::QR::solve(H, -g); // solve H * delta_x = -g
        } catch (const std::runtime_error& e) {
            std::cerr << "Error in solving linear system: " << e.what() << std::endl;
            return x_input; // return the current point if there's an error
        }
        double lambda = 1.0; // initial step size
        double fx = f(x); // function value at current x
        while (lambda >= 1.0/1024){
            double new_fx = f(x + lambda * delta_x); // compute the function value at the new point
            if (new_fx < fx) {break;} // If the function value decreases, accept the step
            lambda /= 2.0; // Else, reduce the step size
        }
        x += lambda * delta_x; // update x with the step size
        current_iter++; // increment the iteration counter
    }
    return x;
}
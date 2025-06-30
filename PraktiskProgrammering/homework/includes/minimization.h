#include"matrix.h"
#include<iostream>
#include<functional>
#include<cmath>

using vec = pp::vector;
using mat = pp::matrix;

vec gradient(std::function<double(const vec&)>, vec&);
mat hessian(std::function<double(const vec&)>, vec&);
vec newton(std::function<double(const vec&)>, vec&, double acc = 1e-6, int max_iter = 1000);

extern int current_iter; // Global variable to count iterations
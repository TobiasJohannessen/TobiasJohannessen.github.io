#include"ODE.h"
#include"../includes/matrix.h"
#include<vector>
#include<cmath>


using vec = pp::vector;
std::tuple<vec, vec> rkstep12(
	std::function<vec(double, const vec&)> f,
	double x,
	const vec& y,
	double dx
){
    vec k0 = f(x,y);
    vec k1 = f(x+dx/2, y+(k0 * dx/2));
    vec yh = y + k1 * dx;
    vec delta_y = (k1-k0)*dx;
    return std::make_tuple(yh, delta_y);
};

/**
 * @brief Integrates an ODE system using an adaptive Runge-Kutta stepper.
 * 
 * @param F      The function representing the ODE system: dy/dx = F(x, y).
 * @param x_init The initial value of the independent variable.
 * @param x_final The final value of the independent variable.
 * @param y_init The initial value of the dependent variable(s).
 * @param dx     The initial step size (default: 0.125).
 * @param acc    The absolute accuracy goal (default: 0.01).
 * @param eps    The relative accuracy goal (default: 0.01).
 * @return A tuple containing a vector of x values and a vector of corresponding y values.
 */
std::tuple<std::vector<double>, std::vector<vec>> driver(
    std::function<vec(double, const vec&)> F,
    double x_init,
    double x_final,
    const vec& y_init,
    double dx = 0.125,
    double acc = 0.01,
    double eps = 0.01
){
    double a;
    double b;
    double x;
    vec y;
    a = x_init; b = x_final; x = a; y = y_init;
    std::vector<double> xlist; xlist.push_back(x);
    std::vector<vec> ylist; ylist.push_back(y);
    if (x >= b - 1e-10) return std::make_tuple(xlist, ylist); // Return when x is sufficiently close to b
while (true) { //while-loop
    if (x >= b) return std::make_tuple(xlist, ylist); //Return when the b is reached
    if (x + dx > b) dx = b - x;
    std::tuple<vec, vec> yhy_tuple = rkstep12(F, x, y, dx);
    vec yh = std::get<0>(yhy_tuple);
    vec delta_y = std::get<1>(yhy_tuple);
    double tol;
    if (b != a) {
        tol = (acc + eps * yh.norm()) * pow(dx / (b - a), 1.0 / 2);
    } else {
        tol = acc + eps * yh.norm(); // fallback if b == a
    }
    double err = delta_y.norm();
    if (err<=tol){ //Accept step
        x+= dx, y = yh;
        xlist.push_back(x);
        ylist.push_back(y);
    }
    if (err > 0) dx *= fmin(pow(tol/err, 0.25) * 0.95, 2);
    else dx*=2;
}
// Add a final return statement for safety
return std::make_tuple(xlist, ylist);
}


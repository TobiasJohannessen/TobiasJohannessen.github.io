#include"../includes/ODE.h"
#include<iostream>
#include<fstream>


//DEFINE DIFFERENTIAL EQUATIONS
// Use: The function signature should be vec f(double x, const vec& y)
// where vec is a vector type defined in matrix.h, and y is the state vector at x.
// The function should return the right-hand side of the ODE system, i.e., dy/dx = f(x, y).
// For second-order ODEs, you can decompose into two first-order ODES.
// The state vector y should contain both the function value and its derivative, e.g., y[0] = y and y[1] = dy/dx. 


//Exponential function
vec f_exp(double x, const vec& y){
    double k = 0.01; // Exponential growth rate
    return y; // dy/dx = k * y
};

// Harmonic oscillator function
// y[0] = y, y[1] = dy/dx
vec f_harmonic(double x, const vec& y){
    vec res(2);
    double k = 5.0; // Spring constant
    res[0] = y[1]; // dy/dx = y[1]
    res[1] = -k*y[0]; // d^2y/dx^2 = dy[1]/dx = -k*y[0]
    return res;
};

vec f_dampened_oscillator(double x, const vec& y){
    vec res(2);
    double k = 5.0; // Spring constant
    double b = 0.1; // Damping coefficient
    res[0] = y[1]; // dy/dx = y[1]
    res[1] = -k*y[0] - b*y[1]; // d^2y/dx^2 = dy[1]/dx = -k*y[0] - b*y[1]
    return res;
};

vec f_lotka_volterra(double x, const vec& y){
    vec res(2);
    double a = 1.1; // Prey growth rate
    double b = 0.4; // Predation rate
    double c = 0.4; // Predator death rate
    double d = 0.1; // Predator growth rate
    res[0] = a * y[0] - b * y[0] * y[1]; // dy/dx = a*y[0] - b*y[0]*y[1]
    res[1] = -c * y[1] + d * y[0] * y[1] ; // dy/dx = c*y[0]*y[1] - d*y[1]
    return res;
};

double epsilon = 0.01;

vec f_relativistic_orbit(double x, const vec& y){
    vec res(2);
    res[0] = y[1]; // dy/dx = y[1]
    res[1] = 1 - y[0] + epsilon * y[0] * y[0]; // d^2y/dx^2 = dy[1]/dx = 1 - y[0] + epsilon * (y[0])^2 
    return res;
};


int main(){
    double x_init, x_final;
    x_init = 0.1; x_final = 20;

    //EXPONENTIAL FUNCTION

    vec y_init = {1.0}; // Initial condition for the exponential function

    auto [xs, ys] =  driver(f_exp, x_init, x_final, y_init, 0.05, 0.01, 0.01);

    // Print the results
    std::ofstream output_file("data/exponential.txt");
    if (!output_file) {
        std::cerr << "Error: cannot open file exponential.txt" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < xs.size(); ++i) {
        output_file << xs[i] << " ";
        for (int j = 0; j < ys[i].size(); ++j) {
            output_file << ys[i][j] << " ";
        }
        output_file << "\n";
    }
    output_file.close();


    // HARMONIC OSCILLATOR

    vec y_init_harmonic = {3.0, -1.0};
    auto [xs_harmonic, ys_harmonic] = driver(f_harmonic, x_init, x_final, y_init_harmonic, 0.05, 0.01, 0.01);

    // Print the results for the harmonic oscillator
     //Output the interpolated values for one constant y-index
    std::ofstream output_harmonic("data/harmonic.txt");
    if (!output_harmonic) {
        std::cerr << "Error: cannot open file data/harmonic.txt" << std::endl;
        return 1;
    }
    output_harmonic << "x\t y \t value\n";
    // Write the values for one constant y-index:
    // Interpolate along same y as original constant_y slice
    for (size_t i = 0; i < xs_harmonic.size(); ++i) {
        output_harmonic << xs_harmonic[i] << "\t" << ys_harmonic[i][0] << "\t" << ys_harmonic[i][1] << "\n";
    }
    output_harmonic.close();


    // DAMPENED OSCILLATOR
    vec y_init_dampened = {3.0, -1.0};
    auto [xs_dampened, ys_dampened] = driver(f_dampened_oscillator, x_init, x_final, y_init_dampened, 0.05, 0.01, 0.01);
    // Print the results for the dampened oscillator
    std::ofstream output_dampened("data/dampened_oscillator.txt");
    if (!output_dampened) {
        std::cerr << "Error: cannot open file data/dampened_oscillator.txt" << std::endl;
        return 1;
    }
    output_dampened << "x\t y \t value\n";
    // Write the values for one constant y-index:
    // Interpolate along same y as original constant_y slice
    for (size_t i = 0; i < xs_dampened.size(); ++i) {
        output_dampened << xs_dampened[i] << "\t" << ys_dampened[i][0] << "\t" << ys_dampened[i][1] << "\n";
    }
    output_dampened.close();


    // LOTKA-VOLTERRA SYSTEM
    x_init = 0.0; x_final = 100.0; // Time range for the Lotka-Volterra system
    vec y_init_lotka = {10.0, 10.0}; // Initial conditions for prey and predator populations
    auto [xs_lotka, ys_lotka] = driver(f_lotka_volterra, x_init, x_final, y_init_lotka, 0.05, 0.01, 0.01);
    // Print the results for the Lotka-Volterra system
    std::ofstream output_lotka("data/lotka_volterra.txt");
    if (!output_lotka) {
        std::cerr << "Error: cannot open file data/lotka_volterra.txt" << std::endl;
        return 1;
    }
    output_lotka << "x\t prey \t predator\n";
    // Write the values for prey and predator populations
    for (size_t i = 0; i < xs_lotka.size(); ++i) {
        output_lotka << xs_lotka[i] << "\t" << ys_lotka[i][0] << "\t" << ys_lotka[i][1] << "\n";
    }
    output_lotka.close();


    // RELATIVISTIC ORBIT
    //Test three scenarios

    // 1. Newtonian limit (epsilon = 0), circular orbit
    x_init = 0.0; x_final = 10.0; // Angle theta
    epsilon = 0.0; //Newtonian limit
    vec y_init_newtonian = {1.0, 0.01}; // Initial conditions for the Newtonian orbit
    auto [xs_newtonian, ys_newtonian] = driver(f_relativistic_orbit, x_init, x_final, y_init_newtonian, 0.01, 0.0001, 0.001);
    // Print the results for the Newtonian orbit
    std::ofstream output_newtonian("data/circular_orbit.txt");
    if (!output_newtonian) {
        std::cerr << "Error: cannot open file data/circular_orbit.txt" << std::endl;
        return 1;
    }
    output_newtonian << "x\t y \t value\n";
    // Write the values for the Newtonian orbit
    for (size_t i = 0; i < xs_newtonian.size(); ++i) {
        output_newtonian << xs_newtonian[i] << "\t" << ys_newtonian[i][0] << "\t" << ys_newtonian[i][1] << "\n";
    }
    output_newtonian.close();

    // 2. Newtonian limit (epsilon = 0), elliptical orbit
    x_init = 0.0; x_final = 20.0; // Angle theta
    epsilon = 0.0; //Newtonian limit
    vec y_init_elliptical = {1.0, -0.5};
    auto [xs_elliptical, ys_elliptical] = driver(f_relativistic_orbit, x_init, x_final, y_init_elliptical, 0.05, 0.01, 0.01);
    // Print the results for the elliptical orbit
    std::ofstream output_elliptical("data/elliptical_orbit.txt");
    if (!output_elliptical) {
        std::cerr << "Error: cannot open file data/elliptical_orbit.txt" << std::endl;
        return 1;
    }
    output_elliptical << "x\t y \t value\n";
    // Write the values for the elliptical orbit
    for (size_t i = 0; i < xs_elliptical.size(); ++i) {
        output_elliptical << xs_elliptical[i] << "\t" << ys_elliptical[i][0] << "\t" << ys_elliptical[i][1] << "\n";
    }
    output_elliptical.close();

    // 3. Relativistic orbit (epsilon = 0.01), elliptical orbit
    x_init = 0.0; x_final = 20.0; // Angle theta
    epsilon = 0.01; // Relativistic limit
    vec y_init_relativistic = {1.0, -0.5};
    auto [xs_relativistic, ys_relativistic] = driver(f_relativistic_orbit, x_init, x_final, y_init_relativistic, 0.05, 0.01, 0.01);
    // Print the results for the relativistic orbit
    std::ofstream output_relativistic("data/relativistic_orbit.txt");
    if (!output_relativistic) {
        std::cerr << "Error: cannot open file data/relativistic_orbit.txt" << std::endl;
        return 1;
    }
    output_relativistic << "x\t y \t value\n";
    // Write the values for the relativistic orbit
    for (size_t i = 0; i < xs_relativistic.size(); ++i) {
        output_relativistic << xs_relativistic[i] << "\t" << ys_relativistic[i][0] << "\t" << ys_relativistic[i][1] << "\n";
    }
    output_relativistic.close();




return 0;
}
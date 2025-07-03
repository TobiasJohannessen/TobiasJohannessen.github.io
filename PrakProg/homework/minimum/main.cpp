#include"../includes/matrix.h"
#include"../includes/minimization.h"
#include<iostream>
#include<functional>
#include<cmath>


using vec = pp::vector;
using mat = pp::matrix;





double f_rosenbrock(const vec& x) {
    if (x.size() != 2) {
        throw std::invalid_argument("Rosenbrock function requires a 2D input.");
    }
    double a = 1.0, b = 100.0;
    double f1 = a - x[0];
    double f2 = (x[1] - x[0] * x[0]);
    double result = f1 * f1 + b * f2 * f2;
    return result;
    // f(x) = (a - x0)^2 + b * (x1 - x0^2)^2, which has a minimum at (1, 1)
}

double f_himmelblau(const vec& x) {
    if (x.size() != 2) {
        throw std::invalid_argument("Himmelblau function requires a 2D input.");
    }
    double x0 = x[0], x1 = x[1];
    double f1 = x0 * x0 + x1 - 11;
    double f2 = x0 + x1 * x1 - 7;
    double result = f1 * f1 + f2 * f2;
    return result;
   
}



double test_newton(std::function<double(vec)> f, vec x0, double acc = 1e-6, std::vector<vec> exp_roots = std::vector<vec>{}, int max_iter = 1000) {
    vec root;
    try {
        root = newton(f, x0, acc, max_iter);
        std::cout << "Root(s) found:\n";
        for (int i = 0; i < root.size(); ++i) {
            std::cout << "x[" << i << "] = " << root[i] << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "Test failed: Newton's method did not converge.\n";
        return 1;
    }

    if (exp_roots.empty()) {
        std::cout << "No expected root provided. Skipping comparison.\n";
        return 0;
    }

    for (size_t i = 0; i < exp_roots.size(); ++i) {
        if (approx(root, exp_roots[i], acc)) {
            std::cout << "Test passed: Root is close to expected value #" << i << ".\n";
            return 0;
        }
    }

    std::cout << "Test failed: No expected root matched within tolerance.\n";
    return 1;
}

int main(){

    std::cout << "Testing Newton's method for minimization...\n" << std::endl;

    // Testing the Rosenbrock function
    std::cout << "\n\n-----------------ROSENBROCK FUNCTION--------------------- \n\n";

    std::cout << "Finding the minima of the Rosenbrock function\n This is done through gradient descent.\n";

    std::cout << "Testing g(x) = (1 - x0, 100 * (x1 - x0^2)), with various starting guesses:" << std::endl;
    int max_iter = 50000;
    int total_count = 0;
    int fail_count = 0;
    for (int i = 0; i < 10; ++i) {
        //random initial guess
        total_count++;
        current_iter = 0;
        
        vec x0_rosenbrock = vec{ static_cast<double>(rand() % 10) , static_cast<double>(rand() % 10) }; // Random initial guess for Rosenbrock
        std::cout << "Initial guess: (" << x0_rosenbrock[0] << ", " << x0_rosenbrock[1] << ")\n";
        fail_count += test_newton(f_rosenbrock, x0_rosenbrock, 1e-4, { vec{ 1.0, 1.0 }}, max_iter); // Expected root at (1, 1)
        std::cout << "Number of steps taken: " << current_iter << "\n\n";
    }
    if (fail_count == 0) {
        std::cout << "All tests passed for the Rosenbrock function.\n";
    } else {
        std::cout << fail_count << " out of " << total_count << " tests failed for the Rosenbrock function.\n";
    }
    
    std::cout << "\n\n -------------------HIMMELBLAU FUNCTION------------------------------- \n\n";


    // Himmelblau function 
    std::cout << "Finding the minima of the Himmelblau function:" << std::endl;
    total_count = 0;
    fail_count = 0;
    for (int i = 0; i < 10; ++i) {
        //random initial guess
        total_count++;
        current_iter = 0;
        vec x0_himmelblau = vec{ static_cast<double>(rand() % 10), static_cast<double>(rand() % 10) }; // Random initial guess for Rosenbrock
        std::vector<vec> root_himmelblau = { vec{ 3.0, 2.0 }, vec{ -2.805118, 3.131312 }, vec{ -3.779310, -3.283186 }, vec{ 3.584428, -1.848126 } }; // Expected roots for Himmelblau function
        // These roots are known for the Himmelblau function
        std::cout << "Initial guess: (" << x0_himmelblau[0] << ", " << x0_himmelblau[1] << ")\n";
        fail_count += test_newton(f_himmelblau, x0_himmelblau, 1e-4, root_himmelblau, max_iter); // Expected root at (1, 1)
        std::cout << "Number of steps taken: " << current_iter << "\n\n";
    }
    if (fail_count == 0) {
        std::cout << "All tests passed for the Himmelblau function.\n";
    } else {
        std::cout << fail_count << " out of " << total_count << " tests failed for the Himmelblau function.\n";
    }


    return 0;



}
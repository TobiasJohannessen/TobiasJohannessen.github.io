#include <cstddef>
#include<vector>
#include<functional>
#include<cmath>
#include"../includes/matrix.h"
#include"../includes/ODE.h"
#include<iostream>
#include<fstream>

using namespace pp;
using vec = vector;
using mat = matrix;

int NSteps = 0;
const double DEFAULT_DX = pow(2, -26); // Default step size for numerical differentiation
mat jacobian(std::function<vec(vec)> f, vec x, vec fx = vec(1), vec dx = vec(1)) {
    if ((dx.size() == 1 && dx[0] == 0.0) || dx.size() != x.size()) {
        dx = vec(x.size()); 
        dx = dx.map([](NUMBER) { return DEFAULT_DX; });
    }
    if (fx.size() != f(x).size()) { fx = f(x); }

    int n = x.size();
    mat J(fx.size(), x.size());
    for (int i = 0; i < n; ++i) {
        vec x_plus = x;
        x_plus[i] += dx[i];
        vec fx_plus = f(x_plus);
        for (int j = 0; j < n; ++j) {
            J(j, i) = (fx_plus[j] - fx[j]) / dx[i];
        }
    }
    return J;
}

vec newton(std::function<vec(vec)> f, vec x0, double acc=1e-2, vec dx = vec(1)) {
    vec x = x0;
    vec fx = f(x), z, fz;
    while (true){
        
        if (fx.norm() < acc){break;}
        mat J = jacobian(f, x, fx, dx); // Calculate Jacobian
        QR::mtuple QRJ = QR::decomp(J);
        mat Q = std::get<0>(QRJ);
        mat R = std::get<1>(QRJ);
        vec Dx = QR::solve(Q,R,-fx); // Solve R*Dx = Q^T * fx
        double lambda = 1.0; // Initial step size
        while (true){
            NSteps++;
            z = x + lambda * Dx; // Update x with step
            fz = f(z); // Evaluate function at new point
            if (fz.norm() < fx.norm()) { // Check if the new point is better
                x = z; // Accept the new point
                fx = fz; // Update fx
                break; // Exit inner loop
            }
            lambda *= 0.5; // Reduce step size if not accepted
            if (lambda < 1e-10) { // Prevent infinite loop
                throw std::runtime_error("Newton's method did not converge.");
            }
        }
    }
    return x;
};


// Function to test the Newton's method


double test_newton(std::function<vec(vec)> f, vec x0, double acc = 1e-6, std::vector<vec> exp_roots = std::vector<vec>{}, vec dx = vec(1)) {
    vec root;
    try {
        root = newton(f, x0, acc, dx);
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


vec f_sqrt2(vec x) {
    return vec{ x[0] * x[0] - 2.0 };
    // f(x) = x^2 - 2, which has a root at +-sqrt(2)
}

vec f_sin(vec x) {
    return vec{ std::sin(x[0]) };
    // f(x) = sin(x), which has roots at multiples of pi
}

vec f_exp_minus_1(vec x) {
    return vec{ std::exp(x[0]) - 1.0 };
    // f(x) = exp(x) - 1, which has a root at x = 0
}


vec f_circle_diag(vec x) {
    double x0 = x[0], x1 = x[1];
    return vec{
        x0 * x0 + x1 * x1 - 4.0,
        x0 - x1
    };
    // f(x) = (x0^2 + x1^2 - 4, x0 - x1), which has roots at (sqrt(2), sqrt(2)) and (-sqrt(2), -sqrt(2)) on the unit circle
}

vec f_rosenbrock(vec x) {
    if (x.size() != 2) {
        throw std::invalid_argument("Rosenbrock function requires a 2D input.");
    }
    double a = 1.0, b = 100.0;
    double first_term = a - x[0];
    double second_term = (x[1] - x[0] * x[0]);
    return vec{first_term * first_term + b * second_term * second_term};
    // f(x) = (1 - x0, 100 * (x1 - x0^2)), which has a minimum at (1, 1)
}

vec g_rosenbrock(vec x) {
    if (x.size() != 2) {
        throw std::invalid_argument("Rosenbrock function requires a 2D input.");
    }
    double a = 1.0, b = 100.0;
    double x_gradient = -2 * (a - x[0]) + 2 * b * (x[1] - x[0] * x[0]) * x[0];
    double y_gradient = 2 * b * (x[1] - x[0] * x[0]);
    return vec{
        x_gradient,
        y_gradient
    };
    // Gradient of the Rosenbrock function
}

vec g_himmelblau(vec x) {
    if (x.size() != 2) {
        throw std::invalid_argument("Himmelblau function requires a 2D input.");
    }
    double x0 = x[0], x1 = x[1];
    double f1 = x0 * x0 + x1 - 11;
    double f2 = x0 + x1 * x1 - 7;
    // Gradient of the Himmelblau function
    double x_gradient = 4 * x0 * f1 + 2 * f2; // Partial derivative with respect to x0
    double y_gradient = 2 * f1 + 4 * x1 * f2; // Partial derivative with respect to x1
    
    return vec{
        x_gradient,
        y_gradient
    };
    // Gradient of the Himmelblau function
}


double rmin = 1e-4, rmax = 8.0;
double acc = 1e-6, eps = 1e-6; // Default accuracy and epsilon for the hydrogen atom problem
vec M_of_E(vec Evec) {
    double E = Evec[0];
    auto F = [E](double r, const vec& y) {
        return vec{ y[1], -2.0 * (E + 1.0/r) * y[0] };
    };
    auto sol = driver(F, rmin, rmax, vec{rmin - rmin*rmin, 1.0 - 2*rmin}, 0.01, acc, eps);
    auto& ylist = std::get<1>(sol);
    vec y_end = ylist.back();
    return vec{ y_end[0] };
}






int main() {
    // Example usage of the Newton's method
    
    std::cout << "Testing Newton's method..." << std::endl;

    std::cout << "Testing f(x) = x^2 - 2, with inital guess 0:" << std::endl;
    vec x0_sqrt2 = vec{ 0.0 }; // Initial guess for sqrt(2)
    std::vector<vec> root_sqrt2 = { vec{ std::sqrt(2.0) }, vec{ -std::sqrt(2.0) } }; // Expected roots at +-sqrt(2)
    test_newton(f_sqrt2, x0_sqrt2, 1e-6, root_sqrt2);

    std::cout << "\nTesting f(x) = sin(x), with initial guess 3.0:" << std::endl;
    vec x0_sin = vec{ 3.0 }; // Initial guess for sin
    std::vector<vec> root_sin = { vec{ 0.0 }, vec{-M_PI}, vec{M_PI} }; // Expected root at 0, 
    test_newton(f_sin, x0_sin, 1e-6, root_sin);

    std::cout << "\nTesting f(x) = exp(x) - 1, with initial guess 10.0:" << std::endl;
    vec x0_exp = vec{ 10.0 }; // Initial guess for exp - 1
    std::vector<vec> root_exp = { vec{ 0.0 } }; // Expected root at 0
    test_newton(f_exp_minus_1, x0_exp, 1e-6, root_exp);


    std::cout << "\nTesting f(x) = (x0^2 + x1^2 - 4, x0 - x1), with initial guess (0.0, 0.0):" << std::endl;
    vec x0_circle = vec{ -1.0, 0.0 }; // Initial guess for circle
    std::vector<vec> root_circle = { vec{ std::sqrt(2.0), std::sqrt(2.0) }, vec{ -std::sqrt(2.0), -std::sqrt(2.0) } }; // Expected roots at (sqrt(2), sqrt(2)) and (-sqrt(2), -sqrt(2))
    test_newton(f_circle_diag, x0_circle, 1e-6, root_circle);

    // Testing the Rosenbrock function
    std::cout << "\n\n ROSENBROCK FUNCTION \n\n";

    std::cout << "Finding the extrema of the Rosenbrock function\n This is done by finding the roots of its gradient.\n";

    std::cout << "Testing g(x) = (1 - x0, 100 * (x1 - x0^2)), with various starting guesses:" << std::endl;
    int total_count = 0;
    int fail_count = 0;
    for (int i = 0; i < 10; ++i) {
        //random initial guess
        total_count++;
        NSteps = 0;
        vec x0_rosenbrock = vec{ static_cast<double>(rand() % 10) + 0.1, static_cast<double>(rand() % 10) + 0.1 }; // Random initial guess for Rosenbrock
        vec dx_rosen = vec{1e-5, 1e-5};
        std::cout << "Initial guess: (" << x0_rosenbrock[0] << ", " << x0_rosenbrock[1] << ")\n";
        fail_count += test_newton(g_rosenbrock, x0_rosenbrock, 1e-6, { vec{ 1.0, 1.0 } }, dx_rosen); // Expected root at (1, 1)
        std::cout << "Number of steps taken: " << NSteps << "\n\n";
    }
    if (fail_count == 0) {
        std::cout << "All tests passed for the Rosenbrock function.\n";
    } else {
        std::cout << fail_count << " out of " << total_count << " tests failed for the Rosenbrock function.\n";
    }
    
    std::cout << "\n\n -------------------HIMMELBLAU FUNCTION------------------------------- \n\n";


    // Himmelblau function 
    std::cout << "Testing g(x) = (1 - x0, 100 * (x1 - x0^2)), with various starting guesses:" << std::endl;
    total_count = 0;
    fail_count = 0;
    for (int i = 0; i < 10; ++i) {
        //random initial guess
        total_count++;
        NSteps = 0;
        vec x0_himmelblau = vec{ static_cast<double>(rand() % 10) + 0.1, static_cast<double>(rand() % 10) + 0.1 }; // Random initial guess for Rosenbrock
        vec dx_himmelblau = vec{1e-2, 1e-2};
        std::vector<vec> root_himmelblau = { vec{ 3.0, 2.0 }, vec{ -2.805118, 3.131312 }, vec{ -3.779310, -3.283186 }, vec{ 3.584428, -1.848126 }, vec{-0.270845, -0.923039} }; // Expected roots for Himmelblau function
        // These roots are known for the Himmelblau function
        std::cout << "Initial guess: (" << x0_himmelblau[0] << ", " << x0_himmelblau[1] << ")\n";
        fail_count += test_newton(g_himmelblau, x0_himmelblau, 1e-6, root_himmelblau, dx_himmelblau); // Expected root at (1, 1)
        std::cout << "Number of steps taken: " << NSteps << "\n\n";
    }
    if (fail_count == 0) {
        std::cout << "All tests passed for the Himmelblau function.\n";
    } else {
        std::cout << fail_count << " out of " << total_count << " tests failed for the Himmelblau function.\n";
    }



    // BOUND STATES OF HYDROGEN ATOM
    std::cout << "\n\n -------------------BOUND STATES OF HYDROGEN ATOM------------------------------- \n\n";
    
    // Find the ground state energy of the hydrogen atom using Newton's method
    vec Emin = newton(M_of_E, vec{-0.5}, 1e-6, vec{1e-3});
    double E0 = Emin[0];
    std::cout << "Found ground-state energy E0 = " << E0 << "\n";

    // Recompute the wavefunction using the found energy E0 
    auto sol = driver([E0](double r, const vec& y){
        return vec{ y[1], -2.0*(E0 + 1.0/r)*y[0] };
    }, rmin, rmax, vec{rmin - rmin*rmin, 1.0 - 2*rmin}, 0.01, 1e-6, 1e-6);

    auto xs = std::get<0>(sol);
    auto ys = std::get<1>(sol);

    std::ofstream hydrogen_output("data/hydrogen_wf.txt");
    hydrogen_output << "#r\tpsi(r)\texact_f0(r)\n";
    for (size_t i = 0; i < xs.size(); ++i)
        hydrogen_output << xs[i] << "\t" << ys[i][0] << "\t" << xs[i] * std::exp(-xs[i]) << "\n";
    hydrogen_output.close();
    std::cout << "Wavefunction data saved to hydrogen_wf.txt, exact f0(r)=r e^{-r} included.\n";

    //Test convergences:
    std::ofstream convergence_rmax("data/convergence_rmax.txt");
    for (double rmax_i = 4; rmax_i <= 10; rmax_i += 0.5) {
        rmax = rmax_i;
        vec Emin = newton(M_of_E, vec{-0.5}, 1e-6, vec{1e-3});
        double E0 = Emin[0];
        convergence_rmax << rmax_i << "\t" << E0 + 0.5 << "\n";
    }
    convergence_rmax.close();
    rmax = 8.0; // Reset rmax for the next test
    std::ofstream convergence_rmin("data/convergence_rmin.txt");
    for (double rmin_i = 1e-4; rmin_i <= 1.5; rmin_i *= sqrt(10)) {
        rmin = rmin_i;
        vec Emin = newton(M_of_E, vec{-0.5}, 1e-6, vec{1e-3});
        double E0 = Emin[0];
        convergence_rmin << rmin_i << "\t" << E0 + 0.5 << "\n";
    }
    convergence_rmin.close();
    rmin = 1e-4; // Reset rmin for the next test
    std::ofstream convergence_acc("data/convergence_acc.txt");
    for (double acc_i = 1e-6; acc_i <= 1e-2; acc_i *= sqrt(10)) {
        acc = acc_i;
        vec Emin = newton(M_of_E, vec{-0.5}, acc_i, vec{1e-3});
        double E0 = Emin[0];
        convergence_acc << acc_i << "\t" << E0 + 0.5 << "\n";
    }
    convergence_acc.close();
    acc = 1e-6; // Reset acc for the next test
    std::ofstream convergence_eps("data/convergence_eps.txt");
    for (double eps_i = 1e-6; eps_i <= 1e-2; eps_i *= sqrt(10)) {
        eps = eps_i;
        vec Emin = newton(M_of_E, vec{-0.5}, 1e-6, vec{1e-3});
        double E0 = Emin[0];
        convergence_eps << eps_i << "\t" << E0 + 0.5 << "\n";
    }
    convergence_eps.close();

    
    return 0;
};
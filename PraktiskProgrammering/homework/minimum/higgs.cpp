#include"../includes/minimization.h"
#include"../includes/matrix.h"
#include<iostream>
#include<cmath>
#include<fstream>
#include <string>



using vec = pp::vector;

vec E;
vec sigma;
vec dsigma;
double breit_wigner(const double E, const double m, const double gamma, const double A) {
    double result = A / ((E-m) * (E-m) + (gamma * gamma) / 4.0);
    return result;
}

double breit_wigner_deviation(const vec& mga){
    if (mga.size() != 3) {
        throw std::invalid_argument("Input vector must contain exactly three elements: m, gamma, and A.");
    }
    double m = mga[0];
    double gamma = mga[1];
    double A = mga[2];
    double sum = 0.0;
    for (int i = 0; i < E.size(); ++i) {
        double bw = breit_wigner(E[i], m, gamma, A);
        double diff = bw - sigma[i];
        sum += diff * diff / (dsigma[i] * dsigma[i]);
        }
    return sum;
};
    
int main(){

    std::ifstream file("data/higgs.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    };
    std::string line;
    std::getline(file, line); // Skip the header line
    while (std::getline(file, line)) {
        double e, s, ds;
        if (sscanf(line.c_str(), "%lf %lf %lf", &e, &s, &ds) == 3) {
            E.push_back(e);
            sigma.push_back(s);
            dsigma.push_back(ds);
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }
    file.close();
    

    std::cout << "Find the Higgs Width using minimization\n" << std::endl;

    vec mga = vec{ 125.0, 4.5, 25}; // Initial guess for m, gamma, and A
    double acc = 1e-4; // Accuracy for convergence
    int max_iter = 1000; // Maximum number of iterations
    vec root;
    try {
        current_iter = 0; // Reset the iteration counter
        root = newton(breit_wigner_deviation, mga, acc, max_iter);
        std::cout << "Root found:\n";
        std::cout << "m = " << root[0] << ", gamma = " << fabs(root[1]) << ", A = " << root[2] << std::endl;
        std::cout << "Minimum chi^2: " << breit_wigner_deviation(root) << std::endl;
        std::cout << "Number of iterations: " << current_iter << std::endl;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1; // Return an error code if Newton's method fails
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return 1; // Return an error code if the input is invalid
    }

    //generate output file
    std::ofstream output("data/higgs_fit.txt");
    if (!output.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1; // Return an error code if the output file cannot be opened
    }
    output << "#E\t breit_wigner\n";
     
    for (double i = E[0]; i < E[E.size() - 1]; i+= 0.1) {
        double bw = breit_wigner(i, root[0], fabs(root[1]), root[2]);
        output << i << "\t" << bw << "\n";
    }
    output.close();

    std::cout << "I seem to consistently get a Higgs Width that is too low (~2.08633). It is also the solution that gives the best chi^2, so I guess it's good?" << std::endl;

    return 0; // Return 0 to indicate success
}
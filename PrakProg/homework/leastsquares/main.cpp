#include "../includes/matrix.h"
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

using mat = pp::matrix;
using vec = pp::vector;

// Return coefficients and covariance matrix
std::pair<vec, mat> lsfit(std::vector<std::function<double(double)>> f, std::vector<double> x, std::vector<double> y, std::vector<double> dy) {
    if (x.size() != y.size() || x.size() != dy.size())
        throw std::invalid_argument("Input vectors x, y, and dy must have the same size.");

    int n = x.size();
    int m = f.size();
    mat A(n, m);
    vec b(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            A(i, j) = f[j](x[i]) / dy[i];
        b[i] = y[i] / dy[i];
    }

    vec coefficients = pp::QR::solve(A, b);
    mat R = std::get<1>(pp::QR::decomp(A));
    mat R_inv = pp::QR::inverse(R);
    mat cov_matrix = R_inv * R_inv.transpose();

    return {coefficients, cov_matrix};
}

int main() {
    std::ifstream inputFile("data/decay.txt");
    if (!inputFile) {
        std::cerr << "Error opening file.\n";
        return 1;
    }

    std::ofstream outputFile("data/decay_log.txt");
    outputFile << "# t\tlog(y)\tdlny\n";

    vec x, y, dy;
    double xi, yi, dyi;
    std::string header;
    std::getline(inputFile, header); // Skip header

    for (std::string line; std::getline(inputFile, line);) {
        if (sscanf(line.c_str(), "%lf %lf %lf", &xi, &yi, &dyi) == 3) {
            x.push_back(xi);
            y.push_back(yi);
            dy.push_back(dyi);
            outputFile << xi << "\t" << std::log(yi) << "\t" << dyi / yi << "\n";
        } else {
            std::cerr << "Error parsing line: " << line << "\n";
        }
    }
    inputFile.close();

    // Transform y -> ln(y), dy -> dy/y
    vec lny = y, dlny = dy;
    for (int i = 0; i < y.size(); ++i) {
        if (y[i] <= 0 || dy[i] <= 0) {
            std::cerr << "Non-positive y or dy value.\n";
            return 1;
        }
        lny[i] = std::log(y[i]);
        dlny[i] = dy[i] / y[i];
    }

    auto fs = std::vector<std::function<double(double)>>{
        [](double z) { return 1.0; },
        [](double z) { return -z; }
    };

    auto [coeffs, cov] = lsfit(fs, x.data, lny.data, dlny.data);

    double lnA = coeffs[0];
    double lambda = coeffs[1];
    double sigma_lnA = std::sqrt(cov(0, 0));
    double sigma_lambda = std::sqrt(cov(1, 1));

    double A = std::exp(lnA);
    double sigma_A = A * sigma_lnA;  // Propagation: σ_A = exp(lnA) * σ_lnA

    double half_life = std::log(2) / lambda;
    double sigma_half_life = std::log(2) / (lambda * lambda) * sigma_lambda;


    std::cout << "Fitted coefficients found using linear squars and uncertanties as the square root of the diagonal of the covariance matrix:\n";
    std::cout << "Fitted ln(A) = " << lnA << " ± " << sigma_lnA << "\n";
    std::cout << "Fitted lambda = " << lambda << " ± " << sigma_lambda << "\n";
    std::cout << "A = " << A << " ± " << sigma_A << "\n";
    std::cout << "Half-life: " << half_life << " ± " << sigma_half_life << " days\n";
    std::cout << "The modern value of the half-life of ThX (Ra-224) is 3.6316(14) d, which is not in agreement with the value found here.\n";
    std::cout << "The discrepancy is on the order of " << (half_life - 3.6316) / sigma_half_life << "σ.\n";
    // Write fit
    std::ofstream fitFile("data/decay_log_fit.txt");
    std::ofstream fitFile2("data/decay_fit.txt");
    fitFile << "# t\tfit_ln(y)\n";
    fitFile2 << "# t\tfit(y)\n";
    for (double t = 0; t <= 15; t += 0.01) {
        double ln_y = lnA - lambda * t;
        double ln_y1 = lnA + sigma_lnA - lambda*t; // The fit with uncertainty in lnA
        double ln_y2 = lnA - sigma_lnA - lambda*t; // The fit with uncertainty in lnA
        double ln_y3 = lnA - (lambda + sigma_lambda)*t; // The fit with uncertainty in lambda
        double ln_y4 = lnA - (lambda - sigma_lambda)*t; // The fit with uncertainty in lambda
        fitFile << t << "\t" << ln_y << "\t" << ln_y1 << "\t" << ln_y2 << "\t" << ln_y3 << "\t" << ln_y4 << "\n";
        fitFile2 << t << "\t" << std::exp(ln_y) << "\t" << std::exp(ln_y1) << "\t" << std::exp(ln_y2) << "\t" << std::exp(ln_y3) << "\t" << std::exp(ln_y4) << "\n";
    }

    std::ofstream coeffFile("data/decay_log_coefficients.txt");
    std::ofstream coeffFile2("data/decay_coefficients.txt");
    coeffFile << lnA << "\t" << sigma_lnA << "\t" << lambda << "\t" << sigma_lambda << "\n";
    coeffFile2 << A << "\t" << sigma_A << "\t" << lambda << "\t" << sigma_lambda << "\n";

    return 0;
}

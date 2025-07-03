#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <limits>
#include <functional>
#include<fstream>


// Integration function
constexpr long double PI_ = 3.14159265358979323846264338327950288L;
int Ncalls = 0; // Global counter for function calls

// Integration function
long double integrate(std::function<long double(long double)> f, long double a, long double b,
                      long double acc = 0.0001L, long double eps = 0.001L,
                      long double f2 = std::numeric_limits<long double>::quiet_NaN(),
                      long double f3 = std::numeric_limits<long double>::quiet_NaN()) {
    long double h = b - a;

    //Test for infinite interval and handle it using variable transformation:
   
    if (std::isnan(f2) || std::isnan(f3)) {
        f2 = f(a + 2.0L * h / 6.0L);
        f3 = f(a + 4.0L * h / 6.0L);
    }
    Ncalls++;
    long double f1 = f(a + h / 6.0L);
    long double f4 = f(a + 5.0L * h / 6.0L);
    long double Q = (2.0L * f1 + f2 + f3 + 2.0L * f4) * h / 6.0L;
    long double q = (f1 + f2 + f3 + f4) * h / 4.0L;
    long double error = fabsl(Q - q);
    if (error < acc + eps * fabsl(Q)) {
        return Q;
    } else {
        long double mid = (a + b) / 2.0L;
        long double left = integrate(f, a, mid, acc / sqrtl(2.0L), eps, f1, f2);
        long double right = integrate(f, mid, b, acc / sqrtl(2.0L), eps, f3, f4);
        return left + right;
    }
}


long double CC_integrate(std::function<long double(long double)> f, long double a, long double b,
                         long double acc = 0.0001L, long double eps = 0.001L) {
    if (std::isinf(a) && std::isinf(b)) {
        std::function<long double(long double)> f_transformed = [f](long double t) {
            long double x = tanl(PI_ * (t - 0.5L));
            long double dx_dt = PI_ / cosl(PI_ * (t - 0.5L)) / cosl(PI_ * (t - 0.5L));
            return f(x) * dx_dt;
        };
        return integrate(f_transformed, 0.0L, 1.0L, acc, eps);
    } else if (std::isinf(b)) {
        std::function<long double(long double)> f_transformed = [f, a](long double t) {
            long double x = a + t / (1.0L - t);
            long double dx_dt = 1.0L / (1.0L - t) / (1.0L - t);
            return f(x) * dx_dt;
        };
        return integrate(f_transformed, 0.0L, 1.0L, acc, eps);
    } else if (std::isinf(a)) {
        std::function<long double(long double)> f_transformed = [f, b](long double t) {
            long double x = b - (1.0L - t) / t;
            long double dx_dt = 1.0L / (t * t);
            return f(x) * dx_dt;
        };
        return integrate(f_transformed, 0.0L, 1.0L, acc, eps);
    } else {
        std::function<long double(long double)> f_transformed = [f, a, b](long double x) {
            return f((a + b) / 2.0L + (b - a) / 2.0L * cosl(x)) * sinl(x) * (b - a) / 2.0L;
        };
        return integrate(f_transformed, 0.0L, PI_, acc, eps);
    }
}
// Test functions
long double f_sqrt(long double x)        { return std::sqrt(x); }
long double f_inv_sqrt(long double x)    { return 1.0 / std::sqrt(x); }
long double f_sqrt_2(long double x)      { return std::sqrt(1 - x * x); }
long double f_ln_sqrt(long double x)     { return std::log(x) / std::sqrt(x); }
long double f_gaussian(long double x)  { return std::exp(-x * x); }
long double f_cauchy(long double x)    { return 1.0 / (x * x + 1.0); }
long double f_exp_decay(long double x) { return std::exp(-x); }


// Implement error function:

long double f_erf(long double z, long double acc = 0.001L, long double eps = 0.001L) {
    if (z < 0.0L) return -f_erf(-z, acc, eps);
    if (z == 0.0L) return 0.0L;
    if (z > 1.0L) {
        return 1.0L - integrate([z](long double t) {
            return expl(-powl(z + (1.0L - t) / t, 2.0L)) / (t * t);
        }, 0.0L, 1.0L, acc, eps) * (2.0L / sqrtl(PI_));
    } else {
        return integrate([](long double t) {
            return expl(-t * t);
        }, 0.0L, z, acc, eps) * (2.0L / sqrtl(PI_));
    }
}

int main() {

    // Part A) Integration tests and error function
    std::vector<bool> test_results;
    bool all_tests_passed = true;

    auto run_test = [&](const std::string& name, long double result, long double expected, long double tol = 1e-4) {
        bool pass = std::fabs(result - expected) < tol;
        std::cout << std::fixed << std::setprecision(10);
        std::cout << name << ":\n"
                  << "  Result:   " << result << "\n"
                  << "  Expected: " << expected << "\n"
                  << "  " << (pass ? "PASSED" : "FAILED") << "\n\n";
        test_results.push_back(pass);
        if (!pass) all_tests_passed = false;
    };

    std::cout << "--- Starting Integration Tests ---\n\n";
    std::cout << "Testing integration of various functions using recursive adaptive quadrature.\n\n Test results will be compared against known values to a precision of 4 decimals.\n\n";

    run_test("Integral of sqrt(x) from 0 to 1", integrate(f_sqrt, 0.0, 1.0), 2.0 / 3.0);
    run_test("Integral of 1/sqrt(x) from 0 to 1", integrate(f_inv_sqrt, 0.0, 1.0), 2.0);
    run_test("Integral of sqrt(1 - x^2) from 0 to 1", integrate(f_sqrt_2, 0.0, 1.0), PI_ / 4.0);
    run_test("Integral of log(x)/sqrt(x) from 0 to 1", integrate(f_ln_sqrt, 0.0, 1.0), -4.0);

    


    std::cout << "--- Testing Error Function erf(z) ---\n\n";

    //Read table values from file "erf_table.data":
    std::ifstream infile("data/erf_table.data");
    if (!infile) {
        std::cerr << "Error: Could not open data/erf_table.data for reading.\n";
        return 1;
    }
    std::string str;
    long double z, erf_z, inv_erf_z;
    std::cout << "Reading values from data/erf_table.data:\n";
   
    
    std::ofstream outfile("data/erf_results.txt");
    std::cout << "z \t table erf(z) \t computed erf(z) \t absolute error \n";
    outfile << "z \t table erf(z) \t computed erf(z) \t absolute error \n";
    if (!outfile) {
        std::cerr << "Error: Could not open data/erf_results.txt for writing.\n";
        return 1;
    }
    while (infile >> z >> erf_z >> inv_erf_z) {
        
        long double computed_erf = f_erf(z);
        std::cout << std::fixed << std::setprecision(10)
                  << z << "\t" << erf_z << "\t" << computed_erf << "\t" << fabsl(erf_z-computed_erf) <<"\n";
        outfile << std::fixed << std::setprecision(10)
                << z << "\t" << erf_z << "\t" << computed_erf << "\t" << fabsl(erf_z-computed_erf) << "\n";
                }
    infile.close();
    outfile.close();
    std::cout << "Finished reading values from data/erf_table.data and writing results to data/erf_results.txt.\n\n";


    // Test accuracy of implementation:

    constexpr long double erf_1 = 0.842700792949714869341220635082609L;

    std::ofstream accuracy_file("data/erf_accuracy.txt");
    if (!accuracy_file) {
        std::cerr << "Error: Could not open data/erf_accuracy.txt for writing.\n";
        return 1;
    }

    accuracy_file << std::fixed << std::setprecision(20);
    accuracy_file << "acc\tAbsolute Error\n";

    for (long double acc = 1e-10L; acc <= 1.0L; acc *= 10.0L) {
        long double computed_erf_1 = f_erf(1.0L, acc, acc);
        long double abs_error = fabsl(computed_erf_1 - erf_1);
        accuracy_file << acc << "\t" << abs_error << "\n";
        std::cout << std::fixed << std::setprecision(20);
        std::cout << "erf(1) with acc=" << acc << ":\n"
                  << "  Computed: " << computed_erf_1 << "\n"
                  << "  Expected: " << erf_1 << "\n"
                  << "  Error:    " << abs_error << "\n\n";
    }

    accuracy_file.close();
    std::cout << "Finished testing erf function with various accuracies.\n";


    // Part B) Variable Transformation Quadratures

    std::cout << "--- Starting Variable Transformation Quadratures Tests ---\n\n";


    std::cout << "Testing number of calls during integration of 1/sqrt(x) from 0 to 1 using variable transformation quadrature.\n\n";
    Ncalls = 0; // Reset function call counter
    // Recall number of function calls for the two integration methods:
    integrate(f_inv_sqrt, 0.0, 1.0);
    int Ncalls_1 = Ncalls; // Store number of function calls for the first method
    std::cout << "Number of function calls for adaptive quadrature: " << Ncalls << "\n";

    Ncalls = 0; // Reset function call counter
    CC_integrate(f_inv_sqrt, 0.0, 1.0);
    int Ncalls_2 = Ncalls; // Store number of function calls for the second method
    std::cout << "Number of function calls for variable transformation quadrature: " << Ncalls_2 << "\n";

    std::cout << "Ratio of function calls (adaptive / variable transformation): "
              << static_cast<double>(Ncalls_1) / Ncalls_2 << "\n\n";

    std::cout << "Testing number of calls during integration of ln(x)/sqrt(x) from 0 to 1 using variable transformation quadrature.\n\n";

    Ncalls = 0; // Reset function call counter
    integrate(f_ln_sqrt, 0.0, 1.0);
    int Ncalls_3 = Ncalls; // Store number of function calls for the first method
    std::cout << "Number of function calls for adaptive quadrature: " << Ncalls_3 << "\n";
    
    Ncalls = 0; // Reset function call counter
    CC_integrate(f_ln_sqrt, 0.0, 1.0);
    int Ncalls_4 = Ncalls; // Store number of function calls for the second method
    std::cout << "Number of function calls for variable transformation quadrature: " << Ncalls_4 << "\n";

    std::cout << "Ratio of function calls (adaptive / variable transformation): "
              << static_cast<double>(Ncalls_3) / Ncalls_4 << "\n\n";


    // Test of infinite intervals using variable transformation quadrature:

    std::cout << "Testing integration of with infinte boundaries\n\n";
    Ncalls = 0; // Reset function call counter
    run_test("Integral of a Gaussian function from -inf to inf",
             CC_integrate(f_gaussian, -std::numeric_limits<long double>::infinity(), std::numeric_limits<long double>::infinity()),
             std::sqrt(PI_));
    std::cout << "Number of function calls for Gaussian integral: " << Ncalls << "\n\n";
    Ncalls = 0; // Reset function call counter
    run_test("Integral of a Cauchy function from -inf to 0",
             CC_integrate(f_cauchy, -std::numeric_limits<long double>::infinity(), 0.0L),
             PI_ / 2.0);
    std::cout << "Number of function calls for Cauchy integral: " << Ncalls << "\n\n";
    Ncalls = 0; // Reset function call counter
    run_test("Integral of a Power Law decay function from 1 to inf",
             CC_integrate(f_exp_decay, 0.0L, std::numeric_limits<long double>::infinity()),
             1.0);
    std::cout << "Number of function calls for Exponential decay integral: " << Ncalls << "\n\n";

    // Summary of test results
     std::cout << "--- Test Summary ---\n";
    if (all_tests_passed) {
        std::cout << "All " << test_results.size() << " integration tests PASSED successfully!\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED. Check the output above for details.\n";
        return 1;
    };

    return 0;
};








   
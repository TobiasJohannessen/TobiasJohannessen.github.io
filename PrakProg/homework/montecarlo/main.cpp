#include <cmath>
#include <functional>
#include <iostream>
#include<random>
#include <vector>
#include <iomanip>
#include<fstream>

std::pair<double, double> montecarlo(std::function<double(std::vector<double>)> f, std::vector<double> a, std::vector<double> b, int N) {
    int dim = a.size();
    double volume = 1.0;
    for (int i = 0; i < dim; ++i) {
        volume *= (b[i] - a[i]);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<std::uniform_real_distribution<>> distributions;
    for (int i = 0; i < dim; ++i) {
        distributions.emplace_back(a[i], b[i]);
    }

    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N; ++i) {
        std::vector<double> point(dim);
        for (int j = 0; j < dim; ++j) {
            point[j] = distributions[j](gen);
        }

        double fx = f(point);
        sum += fx;
        sum2 += fx * fx;
    }

    double mean = sum / N;
    double variance = (sum2 - N * mean * mean) / (N - 1);

    
    return {mean * volume, std::sqrt(variance / N) * volume};
}


double corput(int n, int base = 2) {
    double q = 0.0;
    double bk = 1.0 / base;
    while (n > 0) {
        q += (n % base) * bk;
        n = n / base; //Automatically performs floor division (int/int)
        bk /= base;
    }
    return q;
}

std::vector<int> prime_numbers(int n) {
    std::vector<int> primes;
    for (int i = 2; primes.size() < n; ++i) {
        bool is_prime = true;
        for (int j = 2; j * j <= i; ++j) {
            if (i % j == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            primes.push_back(i);
        }
    }
    return primes;
}
class HaltonSequence {
public:
    HaltonSequence(int dim) {
        if (dim < 1) {
            throw std::invalid_argument("Dimension must be at least 1.");
        }
        bases = prime_numbers(dim);
        cached_points.reserve(1024); // Reserve space to avoid frequent reallocations
    }

    // Get the i-th point in the sequence (0-based)
    std::vector<double> sequence(int n) {
        if (n < 0) throw std::out_of_range("Negative index not allowed.");

        if (n >= static_cast<int>(cached_points.size())) {
            generate_up_to(n + 1); // Generate up to (n+1) points
        }

        return cached_points[n];
    }

    // Precompute the Halton sequence up to `n` points
    void generate_up_to(int n) {
        int start = cached_points.size(); // Don't recompute existing points
        for (int i = start; i < n; ++i) {
            std::vector<double> point(bases.size());
            for (size_t j = 0; j < bases.size(); ++j) {
                point[j] = corput(i, bases[j]);
            }
            cached_points.push_back(point);
        }
    }

    // Optional: access entire cached sequence
    const std::vector<std::vector<double>>& get_cached_sequence() const {
        return cached_points;
    }

    std::pair<double, double> halton_mc(std::function<double(std::vector<double>)> f, std::vector<double> a, std::vector<double> b, int N) {
        double volume = 1.0;
        for (size_t i = 0; i < a.size(); ++i) {
            volume *= (b[i] - a[i]);
        }

        double sum = 0.0, sum2 = 0.0;
        for (int i = 1; i < N; ++i) {
            auto point = sequence(i);
            for (size_t j = 0; j < point.size(); ++j) {
                point[j] = a[j] + point[j] * (b[j] - a[j]);
            }
            double fx = f(point);
            sum += fx;
            sum2 += fx * fx;
        }

        double mean = sum / (N-1);
        double variance = (sum2 - (N-1) * mean * mean) / (N - 2);

        
        return {mean * volume, std::sqrt(variance / (N - 1)) * volume};
    }

private:
    std::vector<int> bases;
    std::vector<std::vector<double>> cached_points;
};



class AdditiveLattice {
    public:
    AdditiveLattice(int dim) {
        if (dim < 1) {
            throw std::invalid_argument("Dimension must be at least 1.");
        }
        alphas = get_alphas(dim);
        cached_points.reserve(1024); // Reserve space to avoid frequent reallocations
    }

    // Get the i-th point in the sequence (0-based)
    std::vector<double> sequence(int n) {
        if (n < 0) throw std::out_of_range("Negative index not allowed.");

        if (n >= static_cast<int>(cached_points.size())) {
            generate_up_to(n + 1); // Generate up to (n+1) points
        }

        return cached_points[n];
    }
    std::vector<double> get_alphas(int n){
        std::vector<int> primes = prime_numbers(n);
        std::vector<double> alphas(primes.size());
        for (size_t i = 0; i < primes.size(); ++i) {
            alphas[i] = std::fmod(std::sqrt(static_cast<double>(primes[i])), 1.0);
        }
        return alphas;
    }
    // Precompute the Halton sequence up to `n` points
    void generate_up_to(int n) {
        int start = cached_points.size(); // Don't recompute existing points
        for (int i = start; i < n; ++i) {
            std::vector<double> point(alphas.size());
            for (size_t j = 0; j < alphas.size(); ++j) {
                point[j] = std::fmod(static_cast<double>(i) * alphas[j], 1.0);
            }
            cached_points.push_back(point);
        }
    }

    // Optional: access entire cached sequence
    const std::vector<std::vector<double>>& get_cached_sequence() const {
        return cached_points;
    }

    std::pair<double, double> lattice_mc(std::function<double(std::vector<double>)> f, std::vector<double> a, std::vector<double> b, int N) {
        double volume = 1.0;
        for (size_t i = 0; i < a.size(); ++i) {
            volume *= (b[i] - a[i]);
        }

        double sum = 0.0, sum2 = 0.0;
        for (int i = 0; i < N; ++i) {
            auto point = sequence(i);
            for (size_t j = 0; j < point.size(); ++j) {
                point[j] = a[j] + point[j] * (b[j] - a[j]);
            }
            double fx = f(point);
            sum += fx;
            sum2 += fx * fx;
        }

        double mean = sum / N;
        double variance = (sum2 - N * mean * mean) / (N - 1);

        
        return {mean * volume, std::sqrt(variance / N) * volume};
    }


private:
    std::vector<double> alphas;
    std::vector<std::vector<double>> cached_points;

};

std::pair<double,double> halton_mc(std::function<double(std::vector<double>)> f, std::vector<double> a, std::vector<double> b, int N) {
    HaltonSequence halton(a.size());
    double volume = 1.0;
    for (size_t i = 0; i < a.size(); ++i) {
        volume *= (b[i] - a[i]);
    }

    double sum = 0.0, sum2 = 0.0;
    for (int i = 1; i < N; ++i) {
        auto point = halton.sequence(i);
        for (size_t j = 0; j < point.size(); ++j) {
            point[j] = a[j] + point[j] * (b[j] - a[j]);
        }
        double fx = f(point);
        sum += fx;
        sum2 += fx * fx;
    }

    double mean = sum / N;
    double variance = (sum2 / N) - (mean * mean);
    
    return {mean * volume, std::sqrt(variance / N) * volume};
}



//Test integrals:
double unit_circle(std::vector<double> p){
    double x = p[0];
    double y = p[1];
    return (x * x + y * y <= 1.0) ? 1.0 : 0.0; // Indicator function for unit circle
}


double difficult_integral(std::vector<double> p) {
    double x = p[0];
    double y = p[1];
    double z = p[2];
    double denominator = 1.0 - cos(x) * cos(y) * cos(z);
    return 1.0 / (denominator * M_PI * M_PI * M_PI);
}
int main(){

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
    // Define the integration limits for a 2D unit square

    std::cout << "--------------------TESTING MONTE CARLO INTEGRATION-----------------:\n";
    std::cout << "Estimating the area of a unit circle using Monte Carlo method.\n\n";
    std::vector<double> a = {0.0, 0.0};
    std::vector<double> b = {1.0, 1.0};

    
    for (int N= 1000; N <= 100000; N *= 10) {
        // Perform Monte Carlo integration
        auto result = montecarlo(unit_circle, a, b, N);

        // Output the result
        std::cout << "N = " << N << ": Estimated integral = " << result.first 
                  << ", Estimated error = " << result.second << std::endl;
        run_test("How close?", result.first, M_PI / 4.0, 1e-2);
    }
    //Gather data points to estimate error function:
    std::ofstream results_file("data/circle_errors.txt");
    for (int N = 1000; N <= 1000000; N += N/10) {
        auto result = montecarlo(unit_circle, a, b, N);
        results_file << N << "\t" << result.first << "\t" << result.second << "\n";
        
    }

    // Test the difficult integral

    std::cout << "\n--------------------TESTING DIFFICULT INTEGRAL-----------------:\n";
    double difficult_integral_result = 1.3932039296856768591842462603255;
    std::vector<double> a_difficult = {0.0, 0.0, 0.0};
    std::vector<double> b_difficult = {M_PI, M_PI, M_PI};
    for (int N = 1000; N <= 1000000; N *= 10) {
        auto result_difficult = montecarlo(difficult_integral, a_difficult, b_difficult, N);
        std::cout << "N = " << N << ": Estimated integral = " << result_difficult.first 
                  << ", Estimated error = " << result_difficult.second << std::endl;
        run_test("Difficult integral", result_difficult.first, difficult_integral_result, 1e-2);
    }
    

    // HALTON SEQUENCE TESTING
    std::cout << "\n--------------------TESTING HALTON SEQUENCE-----------------:\n";
    HaltonSequence halton(2);
    std::cout << "Generating Halton sequence points:\n";
    std::ofstream halton_file("data/halton_sequence.txt");
    if (!halton_file.is_open()) {
        std::cerr << "Error opening file for Halton sequence output.\n";
        return 1;
    }
    halton_file << std::fixed << std::setprecision(10);
    halton_file << "x\t\ty\n";
    for (int i = 0; i < 1000; ++i) {
        auto point = halton.sequence(i);
        halton_file << point[0] << "\t" << point[1] << "\n";
    }
    halton_file.close();

    std::ofstream halton_circle_file("data/halton_errors.txt");
    if (!halton_circle_file.is_open()) {
        std::cerr << "Error opening file for Halton circle output.\n";
        return 1;
    }
    halton_circle_file << std::fixed << std::setprecision(10);
    halton_circle_file << "N\tEstimated Area\tEstimated Error\n";

    HaltonSequence halton_circle(2);
    halton_circle.generate_up_to(1000000); // Precompute Halton sequence points

    for (int N = 1000; N <= 1000000; N += N/10) {
        auto result_halton = halton_circle.halton_mc(unit_circle, a, b, N);
        halton_circle_file << N << "\t" << result_halton.first << "\t" << result_halton.second << "\n";
        //run_test("Halton circle estimation", result_halton.first, M_PI / 4.0, 1e-2);
    }
    halton_circle_file.close();


    // ADDITIVE LATICE TESTING
    std::cout << "\n--------------------TESTING ADDITIVE LATICE-----------------:\n";
    AdditiveLattice lattice(2);
    std::cout << "Generating Additive Lattice points:\n";
    std::ofstream lattice_file("data/additive_lattice.txt");
    if (!lattice_file.is_open()) {
        std::cerr << "Error opening file for Additive Lattice output.\n";
        return 1;
    }
    lattice_file << std::fixed << std::setprecision(10);
    lattice_file << "x\t\ty\n";
    for (int i = 0; i < 1000; ++i) {
        auto point = lattice.sequence(i);
        lattice_file << point[0] << "\t" << point[1] << "\n";
    }
    lattice_file.close();   
    std::ofstream lattice_circle_file("data/lattice_errors.txt");
    if (!lattice_circle_file.is_open()) {
        std::cerr << "Error opening file for Additive Lattice circle output.\n";
        return 1;
    }
    lattice_circle_file << std::fixed << std::setprecision(10);
    lattice_circle_file << "N\tEstimated Area\tEstimated Error\n";
    AdditiveLattice lattice_circle(2);
    lattice_circle.generate_up_to(1000000); // Precompute Additive Lattice points
    for (int N = 1000; N <= 1000000; N += N/10) {
        auto result_lattice = lattice_circle.lattice_mc(unit_circle, a, b, N);
        lattice_circle_file << N << "\t" << result_lattice.first << "\t" << result_lattice.second << "\n";
        //run_test("Additive Lattice circle estimation", result_lattice.first, M_PI / 4.0, 1e-2);
    }
    lattice_circle_file.close();
    return 0;
}
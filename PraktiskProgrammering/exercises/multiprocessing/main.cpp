#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <thread>

struct dataclass {
    public: 
        int start, end;
        double sum;
        char padding[64];
};

double harm(dataclass& dc) { // Pass by reference
    dc.sum = 0;
    for (int i = dc.start; i <= dc.end; i++) {
        dc.sum += 1.0 / i;
    }
    return dc.sum;
}

int main(int argc, char* argv[]) {
    int nthreads = 1, nterms = static_cast<int>(1e8); // Default values

    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-terms" && i + 1 < argc) {
            nterms = static_cast<int>(std::round(std::stod(argv[i + 1])));
        }
        if (arg == "-threads" && i + 1 < argc) {
            nthreads = std::atoi(argv[i + 1]);
        }
        
    }

    std::cout << "Number of threads: " << nthreads << std::endl; // Debugging
    std::cout << "Number of terms: " << nterms << std::endl; // Debugging

    // Initialize thread parameters
    std::vector<dataclass> parameters(nthreads);
    std::vector<std::thread> threads;

    for (int i = 0; i < nthreads; i++) {
        parameters[i].start = 1 + nterms / nthreads * i;
        parameters[i].end = nterms / nthreads * (i + 1);
        threads.emplace_back(harm, std::ref(parameters[i])); // Corrected
    };
    // Join all threads
    for (std::thread& thread : threads) {
        thread.join();
    }

    // Sum up results from all threads
    double sum = 0;
    for (const auto& p : parameters) {
        sum += p.sum;
    }

    std::cout << "main: harmonic sum from " << 1 << " to " << nterms << " = " << sum << std::endl;

    return 0;
}

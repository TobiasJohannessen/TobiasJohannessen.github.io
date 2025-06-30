#include <iostream>
#include <string>
#include <vector>
#include "../includes/matrix.h" // Assuming matrix.h is in ../includes/


// Using namespace pp for convenience as per your matrix.h structure
using namespace pp;

int main(int argc, char* argv[]) {
    int N = 0;

    // Parse command-line arguments to get matrix size N
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("-size:", 0) == 0) { // Check if argument starts with "-size:"
            N = std::stoi(arg.substr(arg.find(":") + 1));
        }
    }

    if (N <= 0) {
        std::cerr << "Error: Please specify a positive matrix size using -size:N\n";
        return 1;
    }

    // Create a random NxN matrix
    matrix A(N, N);
    A.randomize();

    // Perform QR factorization
    // The POSIX 'time' utility will measure the execution time of this part.
    QR::mtuple qr = QR::decomp(A);

    // Optional: Suppress output to cout/cerr for clean timing measurements.
    // If you need to debug, you can uncomment A.print() or other prints.
    // A.print("Random Matrix A:");
    // std::get<0>(qr).print("Q matrix:");
    // std::get<1>(qr).print("R matrix:");

    return 0;
}
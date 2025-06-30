#include <iostream>
#include <vector> // For std::vector
#include <string> // For std::string
#include "../includes/matrix.h" // Assuming matrix.h is in the includes directory
#include <cmath>
#include <cstdlib>
#include <iomanip> // For std::setprecision

using namespace pp;

int main() {
    // Vector to store the results of each test (true for pass, false for fail)
    std::vector<bool> test_results;
    bool all_tests_passed = true;

    // Helper lambda to print test results and update the overall status
    auto run_test = [&](const std::string& test_name, bool condition) {
        std::cout << test_name << ": " << (condition ? "True: PASSED" : "False: FAILED") << std::endl;
        test_results.push_back(condition);
        if (!condition) {
            all_tests_passed = false;
        }
    };

    std::cout << "--- Starting Eigenvalue Decomposition (EVD) Tests ---" << std::endl;

    // --- Test Case 1: Random Symmetric Matrix ---
    std::cout << "\n--- Test Case 1: Random Symmetric Matrix (10x10) ---" << std::endl;
    int dim = 10;
    matrix A = symmetric(dim); // Create a random symmetric matrix
    std::cout << "\nOriginal Symmetric Matrix A (" << dim << " x " << dim << "):" << std::endl;
    A.print();

    
    try {
        EVD evd = EVD(A); // Perform eigenvalue decomposition
        std::cout << "\nEigenvalues (w):" << std::endl;
        evd.w.print();
        std::cout << "\nEigenvectors (V):" << std::endl;
        evd.V.print();

        // 1. Check if V^T * A * V = diag(w)
        std::cout << "\n--- Verification: V^T * A * V == diag(w) ---" << std::endl;
        matrix VTAV = evd.V.T() * A * evd.V;
        VTAV.print("V^T * A * V");

        matrix diag_w = matrix::identity(evd.w.size());
        for (int i = 0; i < evd.w.size(); ++i) {
            diag_w(i, i) = evd.w[i];
        }
        diag_w.print("Diagonal matrix of eigenvalues (diag(w))");
        
        run_test("V^T * A * V is approx. equal to diag(w)", approx(VTAV, diag_w));

        // 2. Check if A = V * D * V^T (reconstruction)
        std::cout << "\n--- Verification: V * diag(w) * V^T == A ---" << std::endl;
        matrix VDVT = evd.V * diag_w * evd.V.T();
        VDVT.print("V * diag(w) * V^T");
        
        run_test("V * diag(w) * V^T is approx. equal to A", approx(VDVT, A));

        // 3. Check if V is orthogonal (V^T * V = I and V * V^T = I)
        std::cout << "\n--- Verification: V is orthogonal (V^T * V = I, V * V^T = I) ---" << std::endl;
        matrix VT_V = evd.V.T() * evd.V;
        VT_V.print("V^T * V");
        matrix V_VT = evd.V * evd.V.T();
        V_VT.print("V * V^T");

        matrix identity_mat = matrix::identity(dim);
        run_test("V^T * V is approx. identity", approx(VT_V, identity_mat));
        run_test("V * V^T is approx. identity", approx(V_VT, identity_mat));

    } catch (const std::runtime_error& e) {
        std::cerr << "\nERROR during EVD calculation or verification: " << e.what() << std::endl;
        // Mark all EVD-related tests as failed if a critical error occurs
        run_test("EVD Calculation (no exception)", false);
        run_test("V^T * A * V is approx. equal to diag(w)", false);
        run_test("V * diag(w) * V^T is approx. equal to A", false);
        run_test("V^T * V is approx. identity", false);
        run_test("V * V^T is approx. identity", false);
    } catch (const std::invalid_argument& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
         // Mark all EVD-related tests as failed if a critical error occurs
        run_test("EVD Calculation (no exception)", false);
        run_test("V^T * A * V is approx. equal to diag(w)", false);
        run_test("V * diag(w) * V^T is approx. equal to A", false);
        run_test("V^T * V is approx. identity", false);
        run_test("V * V^T is approx. identity", false);
    }

    // --- Test Summary ---
    std::cout << "\n--- EVD Test Summary ---" << std::endl;
    if (all_tests_passed) {
        std::cout << "All " << test_results.size() << " EVD checks PASSED successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cout << "Some EVD tests FAILED. Please review the output above for details." << std::endl;
        return 1; // Indicate failure
    }
}
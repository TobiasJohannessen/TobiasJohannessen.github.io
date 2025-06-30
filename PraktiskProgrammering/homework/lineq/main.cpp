#include <iostream>
#include <vector> // For std::vector
#include <string> // For std::string
#include "../includes/matrix.h"
#include <cmath>
#include <cstdlib>
#include <iomanip> // For std::setprecision

using namespace pp;

int main() {

    vector test = vector(10);
    test.print();
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

    std::cout << "--- Starting Matrix and QR Decomposition Tests ---" << std::endl;

    matrix A = matrix(15, 10);
    std::cout << "\nMatrix A (15 x 10)" << std::endl;
    A.randomize();
    A.print();

    // Decompose into Q and R
    QR::mtuple qr = QR::decomp(A);
    matrix Q = std::get<0>(qr); // Q matrix
    matrix R = std::get<1>(qr); // R matrix

    std::cout << "\nMatrix R (should be upper triangular)" << std::endl;
    R.print();
    run_test("R is upper triangular", R.is_upper_triangular());

    std::cout << "\nMatrix product of Q^T Q (Should be identity)" << std::endl;
    matrix QT = Q.transpose();
    matrix QTQ = QT * Q;
    QTQ.print();
    matrix iden_QTQ = matrix(QTQ.size1(), QTQ.size2());
    iden_QTQ.setid();
    run_test("Q^T Q is identity", pp::approx(QTQ, iden_QTQ));

    std::cout << "\nMatrix QR (Should be equal to matrix A)" << std::endl;
    matrix M_QR = Q * R;
    M_QR.print();
    run_test("QR is equal to A", pp::approx(M_QR, A));

    std::cout << "\n--- Testing QR::solve function ---" << std::endl;

    // Check solve function:
    vector x_true(A.size2());
    x_true.randomize();
    vector b = A * x_true; // b = A*x_true
    std::cout << "\nVector b (15 x 1) created from A*x_true (To ensure a solution exists)" << std::endl;
    b.print();
    
    // Attempt to solve Ax = b (using Q and R)
    vector x_solved;
    try {
        x_solved = QR::solve(Q, R, b);
        std::cout << "\nSolution x (R*x should be equal to Q^T b)" << std::endl;
        x_solved.print();

        // Check 1: R*x_solved == Q^T b
        vector check_QTb = Q.transpose() * b;
        vector Rx_solved = R * x_solved;
        std::cout << "\nCheck Q^T b (should be equal to R*x_solved):" << std::endl;
        std::cout << "\nR*x_solved:" << std::endl;
        Rx_solved.print();
        std::cout << "\nQ^T b:" << std::endl;
        check_QTb.print();
        run_test("R*x_solved is equal to Q^T b", pp::approx(Rx_solved, check_QTb));

        // Check 2: A*x_solved == b
        vector check_Ax = A * x_solved;
        std::cout << "\nCheck A*x_solved (should be equal to b):" << std::endl;
        std::cout << "\nA*x_solved:" << std::endl;
        check_Ax.print();
        std::cout << "\nb:" << std::endl;
        b.print();
        run_test("A*x_solved is equal to b", pp::approx(check_Ax, b));
    } catch (const std::runtime_error& e) {
        std::cerr << "\nERROR during QR::solve: " << e.what() << std::endl;
        // Mark solve-related tests as failed if an exception occurs
        run_test("QR::solve execution (no exception)", false);
        run_test("R*x_solved is equal to Q^T b", false); // These tests implicitly fail
        run_test("A*x_solved is equal to b", false); // These tests implicitly fail
    }

    std::cout << "\n--- Testing QR::det and QR::inverse functions ---" << std::endl;

    // Check determinant:
    double det_R = QR::det(R);
    std::cout << "\nDeterminant of R: " << std::fixed << std::setprecision(6) << det_R << std::endl;
    bool det_R_non_singular = std::abs(det_R) > 1e-9; // Use a reasonable threshold for non-zero
    run_test("Determinant of R is non-singular", det_R_non_singular);

    // Check inverse:
    matrix C = matrix(10, 10);
    C.randomize();
    std::cout << "\nMatrix C (10 x 10):" << std::endl;
    C.print();

    matrix C_inv;
    bool inverse_success = true;
    try {
        C_inv = QR::inverse(C);
        std::cout << "\nInverse of C:" << std::endl;
        C_inv.print();
    } catch (const std::invalid_argument& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        inverse_success = false;
        run_test("QR::inverse execution (no exception)", false);
    } catch (const std::runtime_error& e) {
        std::cerr << "\nERROR: " << e.what() << std::endl;
        inverse_success = false;
        run_test("QR::inverse execution (no exception)", false);
    }

    if (inverse_success) {
        matrix C_check = C * C_inv; // Check if C * C_inv is close to identity
        std::cout << "\nCheck C * C_inv (should be close to identity):" << std::endl;
        C_check.print();
        matrix C_iden = matrix(C_check.size1(), C_check.size2());
        C_iden.setid();
        run_test("C * C_inv is close to identity", pp::approx(C_check, C_iden));
    } else {
        // If inverse calculation failed, mark the identity check as failed too
        run_test("C * C_inv is close to identity", false);
    }

    std::cout << "\n--- Test Summary ---" << std::endl;
    if (all_tests_passed) {
        std::cout << "All " << test_results.size() << " checks PASSED successfully!" << std::endl;
        return 0; // Indicate success
    } else {
        std::cout << "Some tests FAILED. Please review the output above for details." << std::endl;
        return 1; // Indicate failure
    }
}
#include"matrix.h"
#include<string>
#include<algorithm>
#include<cmath>
#include<iostream>
#include<cassert>
#include<stdio.h>
#include <random>
#include <iomanip> // For std::fixed, std::setprecision
#include <stdexcept> // For exceptions
#define SELF (*this)
#define FORV(i,v) for(int i=0;i<v.size();i++)
#define FOR_COLS(i,A) for(int i=0;i<A.size2();i++)
namespace pp{

bool approx(NUMBER x,NUMBER y,NUMBER acc=1e-6,NUMBER eps=1e-6){
	if(std::fabs(x-y) < acc)return true;
	if(std::fabs(x-y) < eps*(std::fabs(x)+std::fabs(y)))return true;
	return false;
}

bool approx(const vector& u,const vector& v,NUMBER acc,NUMBER eps){
	if(u.size()!=v.size())return false;
	for(int i=0;i<u.size();i++)if(!approx(u[i],v[i],acc,eps)){
		return false;
	};
	return true;
}


bool approx(const matrix& A,const matrix& B,NUMBER acc,NUMBER eps){
	if(A.size1()!=B.size1() || A.size2() != B.size2())return false;
	for (int i=0; i<A.size2(); i++){
		if (!approx(A[i], B[i], acc, eps)) return false; //Check if each column vector in A is approx-equal to each column vector in B.
	};
	return true;
}

vector& vector::operator+=(const vector& other) {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Vector sizes mismatch for addition.");
    }
    for (size_t i = 0; i < data.size(); ++i) data[i] += other.data[i];
    return *this;
}

vector& vector::operator-=(const vector& other) {
	FORV(i,SELF) data[i]-=other.data[i];
	return SELF; }

vector& vector::operator*=(NUMBER x) {
	FORV(i,SELF) data[i]*=x;
	return SELF; }

vector& vector::operator/=(NUMBER x) {
	FORV(i,SELF) data[i]/=x;
	return SELF; }

vector& vector::add(NUMBER x){
	data.push_back(x);
	return SELF;}

vector& vector::push_back(NUMBER x){
	data.push_back(x);
	return SELF;}

NUMBER vector::norm() const {
    NUMBER s = 0;
    for (size_t i = 0; i < data.size(); ++i) s += data[i] * data[i];
    return std::sqrt(s);
}

double vector::dot(const vector& other){
    if (SELF.size() != other.size()){
        std::cout << "ERROR (Dot Product): The two vectors have incompatible lengths." << std::endl;
        return 0;
    };
    double sum = 0;
    for (int i = 0; i<SELF.size(); i++){
        sum += SELF[i] * other[i];
    };
    return sum;
    };

vector vector::map(std::function<NUMBER(NUMBER)> f) const { // Changed double to NUMBER
    vector r = *this;
    for (int i = 0; i < r.size(); ++i) r[i] = f(r[i]);
    return r;
}

void vector::print(std::string s) const {
    std::cout << s;
    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < data.size(); ++i) std::cout << (double)data[i] << " "; // Cast for output consistency
    std::cout << std::endl;
}

void vector::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<NUMBER> dis(0.0, 1.0); // Example: numbers between 0 and 1

    
	for (int i = 0; i < size(); i++) { // Iterate rows
		(*this)[i] = dis(gen);
	}

}

vector operator/(const vector& v, NUMBER x){
	vector r=v;
	r/=x;
	return r; }

vector operator*(const vector& v, NUMBER x){
	vector r=v;
	r*=x;
	return r; }

vector operator*(NUMBER x,const vector& a){ return a*x; }

vector operator+(const vector& a, const vector& b){
	vector r=a;
	r+=b;
	return r; }

vector operator-(const vector& a){
	vector r=a;
	for(int i=0;i<r.size();i++)r[i]=-r[i];
	return r; }

vector operator-(const vector& a, const vector& b){
	vector r=a;
	r-=b;
	return r; }

NUMBER dot(const vector& v, const vector& w) { // Changed double to NUMBER
    if (v.size() != w.size()) {
        throw std::invalid_argument("Vector sizes mismatch for dot product.");
    }
    NUMBER sum = 0;
    for (int i = 0; i < v.size(); ++i) {
        sum += v[i] * w[i];
    }
    return sum;
}





// MATRIX OPERATIONS AND FUNCTIONS

void matrix::resize(int n, int m){
	cols.resize(m);
	for(int i=0;i<m;++i)cols[i].resize(n);
	}

matrix& matrix::operator+=(const matrix& other) {
    if (size1() != other.size1() || size2() != other.size2()) {
        throw std::invalid_argument("Matrix dimensions mismatch for addition.");
    }
    for (size_t i = 0; i < cols.size(); ++i) cols[i] += other.cols[i];
    return *this;
}

matrix& matrix::operator-=(const matrix& other) {
	FOR_COLS(i,SELF) SELF[i]-=other[i];
	return SELF; }

matrix& matrix::operator*=(NUMBER x) {
	FOR_COLS(i,SELF) SELF[i]*=x;
	return SELF; }

matrix& matrix::operator/=(NUMBER x) {
	FOR_COLS(i,SELF) SELF[i]/=x;
	return SELF; }

matrix operator/(const matrix& A,NUMBER x){
	matrix R=A;
	R/=x;
	return R; }

matrix operator*(const matrix& A,NUMBER x){
	matrix R=A;
	R*=x;
	return R; }

matrix operator*(NUMBER x,const matrix& A){
	return A*x; }

matrix operator+(const matrix& A, const matrix& B){
	matrix R=A;
	R+=B;
	return R; }

matrix operator-(const matrix& A, const matrix& B){
	matrix R=A;
	R-=B;
	return R; }

vector operator*(const matrix& M, const vector& v) {
    if (M.size2() != v.size()) {
        throw std::invalid_argument("Matrix columns must equal vector size for multiplication.");
    }
    vector r; r.resize(M.size1());
    for (int i = 0; i < r.size(); ++i) { // Iterate rows of result
        NUMBER sum = 0;
        for (int j = 0; j < v.size(); ++j) sum += M(i, j) * v[j];
        r[i] = sum;
    }
    return r;
}

matrix operator*(const matrix& A, const matrix& B) {
    if (A.size2() != B.size1()) {
        throw std::invalid_argument("Matrix A columns must equal Matrix B rows for multiplication.");
    }
    matrix R(A.size1(), B.size2());
    for (int i = 0; i < A.size1(); ++i) { // rows of R and A
        for (int j = 0; j < B.size2(); ++j) { // columns of R and B
            for (int k = 0; k < A.size2(); ++k) { // columns of A and rows of B
                R(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return R;
}

matrix symmetric(int dimension) {
    matrix R(dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        for (int j = i; j < dimension; ++j) { // Fill upper triangle and diagonal
            R(i, j) = static_cast<NUMBER>(rand()) / RAND_MAX; // Random value
            R(j, i) = R(i, j); // Mirror to lower triangle
        }
    }
    return R;
}

matrix& matrix::operator*=(const matrix& other) {
    *this = (*this) * other; // Use the global operator*
    return *this;
}

matrix matrix::operator^(int p) {
    if (size1() != size2()) {
        throw std::invalid_argument("Matrix must be square for exponentiation.");
    }
    if (p < 0) {
        throw std::invalid_argument("Negative exponents for matrices are not supported."); // Requires inverse
    }
    matrix result(size1(), size2());
    result.setid(); // Initialize as identity matrix (A^0 = I)

    matrix temp = *this; // Copy of original matrix

    while (p > 0) {
        if (p % 2 == 1) { // If p is odd
            result *= temp;
        }
        temp *= temp; // Square the temp matrix
        p /= 2; // Divide p by 2
    }
    return result;
}

void matrix::setid(){
	assert(size1()==size2());
	for(int i=0;i<size1();i++){
		SELF(i,i)=1;
		for(int j=i+1;j<size1();j++)SELF(i,j)=SELF(j,i)=0;
		}
	}

matrix matrix::transpose() const {
	matrix R(size2(),size1());
	for(int i=0;i<R.size1();i++)
		for(int j=0;j<R.size2();j++) R(i,j)=SELF(j,i);
	return R;
	}

matrix matrix::T() const {return SELF.transpose();}


void matrix::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<NUMBER> dis(0.0, 1.0); // Example: numbers between 0 and 1

    for (int j = 0; j < size2(); j++) { // Iterate columns
        for (int i = 0; i < size1(); i++) { // Iterate rows
            (*this)(i, j) = dis(gen); // Use (row, col) indexing
        }
    }
}

bool matrix::is_upper_triangular() {
    int rows = size1();
    int cols = size2();

    for (int i = 1; i < rows; ++i) {
        for (int j = 0; j < i && j < cols; ++j) { // Iterate lower triangle elements (i > j)
            if (std::abs((*this)(i, j)) > 1e-12)
                return false;
        }
    }
    return true;
}

void matrix::print(std::string s) const {
    std::cout << s << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < size1(); i++) {
        for (int j = 0; j < size2(); j++) std::cout << (double)(*this)(i, j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

matrix matrix::identity(int dimension){
    matrix I(dimension, dimension);
    I.setid(); // Set as identity matrix
    return I;
}



// QR decomposition operations:

QR::mtuple QR::decomp(const matrix& M) {
    matrix Q = M; // Initialize Q with M
    int nrows = M.size1();
    int ncols = M.size2();

    // Ensure M is not empty and is square or tall for QR
    if (ncols == 0 || nrows < ncols) {
        throw std::invalid_argument("Matrix must be square or tall for QR decomposition.");
    }

    matrix R(ncols, ncols); // R is ncols x ncols

    // Gram-Schmidt process
    for (int k = 0; k < ncols; ++k) { // Iterate through columns
        NUMBER col_k_norm = Q.cols[k].norm(); // Norm of the current column
        if (col_k_norm < 1e-12) { // Handle zero column, could indicate singularity
            // Depending on desired behavior, could throw exception or continue
            // For now, let's just make sure we don't divide by zero
            R(k,k) = 0;
            // Optionally, fill the rest of the column with zeros or other value
            continue;
        }
        R(k, k) = col_k_norm;
        Q.cols[k] /= R(k, k); // Normalize current column of Q

        // Orthogonalize remaining columns against the current normalized column
        for (int j = k + 1; j < ncols; ++j) {
            NUMBER dot_prod = dot(Q.cols[k], Q.cols[j]);
            R(k, j) = dot_prod;
            Q.cols[j] -= Q.cols[k] * dot_prod;
        }
    }
    return std::make_tuple(Q, R);
}


vector QR::solve(const matrix& Q, const matrix& R, const vector& b) {
    if (Q.size1() != b.size() || Q.size2() != R.size1() || R.size1() != R.size2()) {
        throw std::invalid_argument("Incompatible dimensions for QR solve.");
    }

    // Solve the system Ax = b where A = QR.
    // We need to solve Q^T b = R x, where Q is orthogonal
    // and R is upper triangular.

    // Step 1: Solve Qy = b => y = Q^T b
    // Q is column-major. Q(row, col) is cols[col][row].
    // (Q^T)_ij = Q_ji = Q.cols[i][j]
    vector y(Q.size2()); // y will have size equal to number of columns of Q
    for (int i = 0; i < Q.size2(); ++i) { // Rows of y (which corresponds to columns of Q)
        NUMBER sum = 0;
        for (int j = 0; j < Q.size1(); ++j) { // Elements of Q^T row and b
            sum += Q(j, i) * b[j]; // Access Q_ji
        }
        y[i] = sum;
    }

    // Step 2: Solve Rx = y using back substitution (R is upper triangular)
    vector x(R.size2());
    for (int i = R.size2() - 1; i >= 0; --i) {
        if (std::abs(R(i,i)) < 1e-12) { // Check for singularity
             throw std::runtime_error("Matrix R is singular, cannot solve.");
        }
        x[i] = y[i];
        for (int j = i + 1; j < R.size2(); ++j) {
            x[i] -= R(i, j) * x[j];
        }
        x[i] /= R(i, i);
    }
    return x;
}

vector QR::solve(const matrix& M, const vector& b) {
    // Perform QR decomposition of M
    QR::mtuple qr = QR::decomp(M);
    matrix Q = std::get<0>(qr);
    matrix R = std::get<1>(qr);

    // Use the solve function with Q and R
    return QR::solve(Q, R, b);
}

double QR::det(const matrix& M) {
    if (M.size1() != M.size2()) {
        throw std::invalid_argument("Determinant is only defined for square matrices.");
    }
    QR::mtuple qr = QR::decomp(M);
    matrix R = std::get<1>(qr);

    NUMBER determinant = 1.0;
    for (int i = 0; i < R.size1(); ++i) {
        determinant *= R(i, i);
    }
    // The sign of the determinant depends on the order of columns or if any reflections happened.
    // For basic Gram-Schmidt without pivots, it usually reflects the sign from the R matrix.
    // However, if M had negative determinant, and Q also accounts for that, the R diagonal product is just the magnitude.
    // For now, let's assume `R` correctly carries the sign.
    return determinant; // Changed from std::fabs to directly return determinant
}

matrix QR::inverse(const matrix& A) {
    if (A.size1() != A.size2()) {
        throw std::invalid_argument("Inverse is only defined for square matrices.");
    }
    QR::mtuple qr = QR::decomp(A);
    matrix Q = std::get<0>(qr);
    matrix R = std::get<1>(qr);

    // Solve Ax_i = e_i for each standard basis vector e_i, where A = QR.
    // The solution x_i will be the i-th column of A_inverse.
    matrix A_inverse(A.size1(), A.size2()); // Initialize A_inverse with correct dimensions
    for (int i = 0; i < A.size2(); ++i) { // Iterate through columns of the identity matrix
        vector e(A.size1()); // e_i will have size of rows of A
        e[i] = 1.0; // Standard basis vector
        
        // QR::solve(Q, R, e) solves QRx = e, which is Ax = e.
        // The returned vector 'x' is the i-th column of A_inverse.
        vector x_col = QR::solve(Q, R, e); 

        // Populate the columns of A_inverse
        for (int j = 0; j < A.size1(); ++j) {
            A_inverse(j, i) = x_col[j]; // A_inverse(row, col) = x_col[row]
        }
    }
    return A_inverse; // This matrix is already A_inverse
}

void EVD::timesJ(matrix& A, int p, int q, NUMBER theta) {
    double c = std::cos(theta);
    double s = std::sin(theta);
    // Loop through all rows of the matrix
    for (int i = 0; i < A.size1(); ++i) {
        // Store original values before modification as they are used in the second calculation
        NUMBER a_ip = A(i, p); // Original A_i,p
        NUMBER a_iq = A(i, q); // Original A_i,q

        // Apply the rotation to the elements in column p
        A(i, p) = c * a_ip - s * a_iq;

        // Apply the rotation to the elements in column q
        A(i, q) = s * a_ip + c * a_iq;
    }
}
void EVD::Jtimes(matrix& A, int p, int q, NUMBER theta) {
    double c = std::cos(theta);
    double s = std::sin(theta);
    // Loop through all columns of the matrix
    for (int j = 0; j < A.size2(); ++j) {
        // Store original values before modification as they are used in the second calculation
        NUMBER a_pj = A(p, j); // Original A_p,j
        NUMBER a_qj = A(q, j); // Original A_q,j

        // Apply the rotation to the elements in row p
        A(p, j) = c * a_pj + s * a_qj;

        // Apply the rotation to the elements in row q
        A(q, j) = -s * a_pj + c * a_qj;
    }
}

EVD::mtuple EVD::cyclic(matrix& A_in, matrix& V_in) {
    int n = A_in.size1(); // Dimension of the square matrix

    // Basic validation for symmetric matrix A
    if (A_in.size1() != A_in.size2()) {
        throw std::invalid_argument("Matrix A must be square for Jacobi diagonalization.");
    }
    // Also, V_in must be square and of the same dimension as A_in
    if (V_in.size1() != n || V_in.size2() != n) {
        throw std::invalid_argument("Matrix V must be a square matrix of the same dimension as A.");
    }


    bool changed;
    int sweeps = 0; // Optional: count sweeps for debugging or limiting iterations
    const int MAX_SWEEPS = 100; // Prevent infinite loops for ill-conditioned matrices

    do {
        changed = false;
        sweeps++;

        // Iterate through all unique off-diagonal pairs (p, q) where p < q
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                // Get the current off-diagonal element and the diagonal elements
                NUMBER apq = A_in(p, q);
                NUMBER app = A_in(p, p);
                NUMBER aqq = A_in(q, q);

                // Calculate the rotation angle theta
                // The problem statement uses Atan2, which maps to std::atan2
                NUMBER theta = 0.5 * std::atan2(2 * apq, aqq - app);

                NUMBER c = std::cos(theta);
                NUMBER s = std::sin(theta);

                // Calculate the new diagonal elements AFTER the full J^T A J transformation
                // These formulas are provided directly in the exercise description
                NUMBER new_app = c * c * app - 2 * s * c * apq + s * s * aqq;
                NUMBER new_aqq = s * s * app + 2 * s * c * apq + c * c * aqq;

                // Convergence check: If diagonal elements change, a rotation was performed
               
                if (new_app != app || new_aqq != aqq) { 
                    changed = true;

                    // Apply rotation to A: A <- A J
                    EVD::timesJ(A_in, p, q, theta);

                    // Apply rotation to A: A <- J^T A
                    EVD::Jtimes(A_in, p, q, -theta);

                    // Apply rotation to V:
                    EVD::timesJ(V_in, p, q, theta);
                }
            }
        }
    } while (changed && sweeps < MAX_SWEEPS); // Continue if changes were made and max sweeps not reached

    // After convergence, A_in should be diagonal (D) and V_in contains eigenvectors (V)
    return std::make_tuple(A_in, V_in);
}

// EVD constructor implementation
EVD::EVD(const matrix& M) {
    if (M.size1() != M.size2()) {
        throw std::invalid_argument("Eigenvalue decomposition is only defined for square matrices.");
    }
    // Make a copy of M to modify it in-place for diagonalization
    matrix A_copy = M;
    // Initialize V as an identity matrix of the same dimension
    V = matrix::identity(M.size1());

    // Perform cyclic Jacobi diagonalization
    // The result tuple contains the diagonalized matrix (D) and the accumulated eigenvectors (V)
    mtuple result = EVD::cyclic(A_copy, V);

    // A_copy is now the diagonal matrix D
    matrix D_matrix = std::get<0>(result);
    // V is the eigenvector matrix (which was passed by reference and modified)
    V = std::get<1>(result);

    // Extract eigenvalues from the diagonal of D_matrix
    w = vector(D_matrix.size1());
    for (int i = 0; i < D_matrix.size1(); ++i) {
        w[i] = D_matrix(i, i);
    }
}
}//pp

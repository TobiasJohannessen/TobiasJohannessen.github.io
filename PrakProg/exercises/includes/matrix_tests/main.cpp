#include <iostream>
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding rand()
#include "../matrix.h"

#define FORV(i,v) for(int i=0;i<v.size();i++)
#define FOR_COLS(i,A) for(int i=0;i<A.size2();i++)

int main() {
    using namespace pp;
    std::srand(std::time(nullptr)); // Seed random number generator
    
    // Create two 4x4 matrices with random integers
    matrix m1(4, 4), m2(4, 4);
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            m1[i][j] = std::rand() % 10;  // Random number [0, 9]
            m2[i][j] = std::rand() % 10;
        }
    }
    
    m1.print("Matrix 1 = ");
    m2.print("Matrix 2 = ");

    // Matrix operations
    matrix m_add = m1 + m2;
    matrix m_sub = m1 - m2;
    matrix m_mult = m1 * m2;
    matrix m_scaled = m1 * 2.0;
    
    std::cout << "Matrix addition: " << std::endl; m_add.print("m1 + m2");
    std::cout << "Matrix subtraction: " << std::endl; m_sub.print("m1 - m2");
    std::cout << "Matrix multiplication: " << std::endl; m_mult.print("m1 * m2");
    std::cout << "Matrix scaled: " << std::endl; m_scaled.print("m1 * 2");
    

    //Non-square matrices and vector products:

    matrix m3(2,4), m4(4 ,2);
    vector v3(4);
    int counter = 0;
    FOR_COLS(i, m3){
        FORV(j, m3[i]){
            counter += 1;
            m3(j,i) = counter;
        } 
    };
    FOR_COLS(i, m4){
        FORV(j, m4[i]){
            m4(j,i) = i + j;
        } 
    };
    FORV(i, v3){
        v3[i] = std::rand() % 10;
    };


    m3.print("Matrix 3"); std::cout << "Number of rows: " << m3.size1() << " Number of columns: " << m3.size2() << std::endl;
    m4.print("Matrix 4");std::cout << "Number of rows: " << m4.size1() << " Number of columns: " << m4.size2() << std::endl; 
    v3.print("Vector 3");
    
    
    
    matrix m_mult2;
    m_mult2 = m3 * m4;
    std::cout << "Product of two non-square matrices:" << std::endl; m_mult2.print("m3 * m4");

    vector v_transformed = m3 * v3;
    std::cout << "Product of a matrix and a vector:" << std::endl; 
    v_transformed.print("m3 * v3");
    
    std::cout << std::endl;
    
    vector v_not_working = m4 * v3;
    v_not_working.print("This should not work:");
    
    return 0;
}

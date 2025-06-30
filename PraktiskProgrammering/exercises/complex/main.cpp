#include<iostream>
#include<complex>
#include<cmath>
#include "../includes/sfuns.h"




int main(){

    //Check sqrt(-1):
    std::complex<double> z1 = -1;
    std::complex<double> z2, z2_expected;
    z2 = sqrt(z1);
    z2_expected = 0 + 1.0j;

    
    std::cout << "sqrt(-1) = " << z2 << " Expected: " << z2_expected << std::endl;
    checkApprox(z2, z2_expected);
    
    //Check ln(i):
    std::complex<double> z3, z3_expected;
    z3 = std::log(z2);
    z3_expected = z2 * M_PI / 2.;
    std::cout << "\nln(i) = " << z3 << " Expected: i*pi/2 = " << z3_expected << std::endl;
    checkApprox(z3, z3_expected);

    //Check sqrt(i)
    std::complex<double> z4, z4_expected;
    z4 = std::sqrt(z2);
    z4_expected = 1/sqrt(2.) + z2 / sqrt(2.);
    std::cout << "\nsqrt(i) = " << z4 << " Expected: 1/sqrt(2) + i/sqrt(2) = " << z4_expected << std::endl;
    checkApprox(z4, z4_expected);
    //Check i^i:
    std::complex<double> z5, z5_expected;
    z5 = std::pow(z2, z2);
    z5_expected = std::exp(-M_PI/2.); 
    std::cout << "\ni^i = " << std::pow(z2,z2) << " Expected: e^(-pi/2) = " << z5_expected << std::endl;
    checkApprox(z4, z4);
  

    double a, b;
    a = 2, b = 2;
    checkApprox(a, b);
    return 0;
}

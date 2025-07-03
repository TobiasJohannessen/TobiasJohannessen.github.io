#ifndef SFUNS_H
#define SFUNS_H

#include <iostream>
#include <complex>

// Math functions:
namespace sfuns{
    double gamma(double x); //gamma function
    double lngamma(double x); //natural log of gamma function
    std::complex<double> cgamma(std::complex<double>);
    double erf(double x); //error function


    //bool approx(double a, double b, double acc = 1e-9, double eps = 1e-9);

    bool approx(std::complex<double> z1, std::complex<double> z2, double acc = 1e-9, double eps = 1e-9);
    void checkApprox(std::complex<double> a, std::complex<double> b);
    }
#endif // SFUNS_H

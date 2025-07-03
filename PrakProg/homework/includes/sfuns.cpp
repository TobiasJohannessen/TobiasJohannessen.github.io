#include<cmath>
#include "sfuns.h" // Include the header file
#include<complex>
#include <cstdlib>

namespace sfuns {

    // Math functions
    double gamma(double x){
        ///single precision gamma function (formula from Wikipedia)
        if(x<0){
            return M_PI/sin(M_PI*x)/gamma(1-x); // Euler's reflection formula
        };
        if(x<9){
            return gamma(x+1)/x; // Recurrence relation;
        };
        double lnfgamma=x*log(x+1/(12*x-1/x/10))-x+log(2*M_PI/x)/2;
        
        return exp(lnfgamma);
    };


    double lngamma(double x){
        ///single precision gamma function (formula from Wikipedia)
        if(x <= 0){ 
            return NAN;
        };
        if(x < 9){ 
            return lngamma(x+1) - std::log(x);
        };
        double lnfgamma=x*log(x+1/(12*x-1/x/10))-x+log(2*M_PI/x)/2;
        
        return lnfgamma;
    };

    std::complex<double> cgamma(std::complex<double>){
        return 0;
    };

    double erf(double x){
        /// single precision error function (Abramowitz and Stegun, from Wikipedia)
        if(x<0) return -erf(-x);
        double a[]={0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429};
        double t=1/(1+0.3275911*x);
        double sum=t*(a[0]+t*(a[1]+t*(a[2]+t*(a[3]+t*a[4]))));/* the right thing */
        return 1-sum*exp(-x*x);
    };

    // Commented out, as this case is included in the complex double version further down. Kept here in case it isn't exactly equal.
    // bool approx(double a, double b, double acc, double eps){
    //     double diff = std::abs(a-b);
    //     double size = std::max(std::abs(a), std::abs(b));
    //     if (diff <= acc){
    //         return true;
    //     };
    //     if (diff/size <= eps){
    //         return true;
    //     };
    //     return false;

    // }

    bool approx(std::complex<double> z1, std::complex<double> z2, double acc, double eps){
        std::complex<double> cdiff = z2 - z1;
        double diff = std::abs(cdiff);
        double size = std::max(std::abs(z1), std::abs(z1));
        if (diff <= acc){
            return true;
        };
        if (diff/size <= eps){
            return true;
        };
        return false;

    }

    void checkApprox(std::complex<double> a, std::complex<double> b){
        std::cout << "Are they (almost) equal? " << (approx(a, b, 1e-9, 1e-9) ? "Yes": "No") << std::endl;
        return;
    }

}
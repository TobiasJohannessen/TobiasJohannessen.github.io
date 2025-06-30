#include <iostream>
#include <string>
#include <cmath>
#include "../includes/sfuns.h"

using namespace std;



const double gamma_values[10] = {
    1.0,             // Γ(1)
    1.0,             // Γ(2)
    2.0,             // Γ(3)
    6.0,             // Γ(4)
    24.0,            // Γ(5)
    120.0,           // Γ(6)
    720.0,           // Γ(7)
    5040.0,          // Γ(8)
    40320.0,         // Γ(9)
    362880.0         // Γ(10)
};

const double log_gamma_values[13] = {
	NAN,             // ln(Γ(0))
    0.0,             // ln(Γ(1))
    0.0,             // ln(Γ(2))
    0.693147,        // ln(Γ(3))
    1.791759,        // ln(Γ(4))
    3.178054,        // ln(Γ(5))
    4.787492,        // ln(Γ(6))
    6.579251,        // ln(Γ(7))
    8.525161,        // ln(Γ(8))
    10.60460,        // ln(Γ(9))
    12.80183,         // ln(Γ(10))
	15.10441,         // ln(Γ(11))
	17.50231,         // ln(Γ(12))
};


int main(){

	cout << "Simple math operations:" << endl;

	
	double sqrt2 = sqrt(2.0);

	cout << "The square root of 2 equals " << sqrt2 << ". Should be 1.41421 "<< endl;

	double root5_2 = pow(2, 1/5.0);

	cout << "The fifth root of 2 equals " << root5_2 << ". Should be 1.1487 " << endl;

	double e_pi = pow(M_E,M_PI);

	cout << "e to the power of pi equals " << e_pi << ". Should be 23.1407 "<< endl;

	double pi_e = pow(M_PI,M_E);

	cout << "pi to the power of pi equals " << pi_e << ". Should be 22.4592 "<< endl;

	cout << endl;

	cout << "Gamma functions of 1-10:" << endl;

	for(int i=1; i<=10; i++){
		cout << "Γ(" << i << ") = " << sfuns::gamma(i) << " --- Real value: " << gamma_values[i-1] << endl;
	};

	cout << endl;

	cout << "ln of Gamma functions of 0-10:" << endl;

	for(int i=0; i<=10; i++){
		cout << "ln(Γ(" << i << ")) = " << sfuns::lngamma(i) << " --- Real value: " << log_gamma_values[i] << endl;
	};


	return 0;
};

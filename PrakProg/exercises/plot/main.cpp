#include <cmath>
#include<iostream>
#include"../includes/sfuns.h"


int main(){
    //std::cout << "Error function" << std::endl;
    double dx = 1.0/128;
    for(double x=-3; x<=3; x+=dx){
        std::cout << x << " " << sfuns::erf(x) << std::endl;
    };
    std::cout << std::endl << std::endl;


    //std::cout << "Gamma function" << std::endl;
    
    for (double x = -5 + dx/2; x <= 5; x+=dx){
        std::cout << x << " " << sfuns::gamma(x) << std::endl;
    };
    std::cout << std::endl << std::endl;
    
    //std::cout << "Ln(Gamma function)" << std::endl;
    for (double x = dx; x <= 10; x+=dx){
        std::cout << x << " " << sfuns::lngamma(x) << std::endl;
    };
    std::cout << std::endl << std::endl;
    
    //std::cout << "Factorials" << std::endl;
    double f = 1;
    std::cout << 1 << " " << f << std::endl;
    for (int i = 1; i<=10; i++){
        f*=i;
        std::cout << i + 1 << " " << f << std::endl;
    };

    std::cout << std::endl << std::endl;

    f = 1;
    std::cout << 1 << " " << std::log(f) << std::endl;
    for (int i = 1; i<=10; i++){
        f*=i;
        std::cout << i + 1 << " " << std::log(f) << std::endl;
    }
    return 0;
}
#include <iostream>
#include <limits>
#include<cmath>
#include <iomanip>
using namespace std;



bool approx(double a, double b, double acc=1e-9, double eps=1e-9){
    double diff = abs(a-b);
    double size = max(abs(a), abs(b));
    if (diff <= acc){
        return true;
    };
    if (diff/size <= eps){
        return true;
    };
    return false;
}

int main()
{

    // Compute the maximum int by already knowing the limit, since overflow doesn't break the loop.

    cout << "Maximum Integer:" << endl;
    int maxInt = 0;
    int i = 0;

    while (i <= numeric_limits<int>::max() - 1)  // Prevent overflow
    {
        i += 1;
    }

    maxInt = i;
    cout << "Max int: " << maxInt << endl << endl;

    cout << "Minimum Integer:" << endl;
    int minInt = 0;
    i = 0;
    // Compute the minimum int by already knowing the limit, since overflow doesn't break the loop.
    while (i >= numeric_limits<int>::min() + 1)  // Prevent overflow
    {
        i -= 1;
    }

    minInt = i;
    cout << "Min int: " << minInt << endl << endl;


    // Machine Epsilon
    cout << "Machine epsilons" << endl;
    double x=1; 
    while(1+x!=1){
        x/=2;
    } 
    x*=2;


    float y=1; 
    while((float)(1+y) != 1)
    {
        y/=2;
    } 
    y*=2;

    cout << "The machine epsilon for doubles is " << x << endl;
    cout << "The machine epsilon for floats is " << y << endl << endl; 

    // Check epsilon properties

    double epsilon= pow(2,-52);
    double tiny=epsilon/2;
    double a=1+tiny+tiny;
    double b=tiny+tiny+1;

    cout << "a = 1 + tiny + tiny" << endl << "b = tiny + tiny + 1" <<endl;
    cout << "a == b: " << (a==b ? "True": "False") << endl;
    cout << "a>1: " << (a>1 ? "True": "False") << endl;
    cout << "b>1: " << (b>1 ? "True": "False") << endl << endl;

    // Comparing doubles
    cout << "Comparing Doubles" << endl;
    cout << "Example:" << endl;

    double d1 = 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1;
    double d2 = 8*0.1;

    std::cout << std::scientific << std::setprecision(15);
    cout << "d1 = " << d1 << endl;
    cout << "d2 = " << d2 << endl;
    cout << "d1 == d2: " << (d1==d2 ? "True": "False") << endl << endl; 

    cout << "Comparing doubles using approximate equality function: " << endl;

    cout << "approx(d1, d2, acc = 1e-9, eps = 1e-9): " << (approx(d1, d2) ? "True": "False") << endl << endl; 

    }

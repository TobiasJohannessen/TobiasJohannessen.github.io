#include "../includes/vec.h"

int main(){

    std::cout << "Initialize some vectors: \n";
    vec v1 = vec(1,2,3);
    vec v2 = vec(3,2,1);
    v1.print("v1 = ");
    v2.print("v2 = ");
    vec v3;
    v3 = v1 + v2;

    std::cout << "\nTest addition: \n";
    v3.print("v3 = v1 + v2 = ");
	v1 += v2;
    v1.print("v1 += v2, v1 = ");

    std::cout << "\nReset v1 \n";
    v1.set(1,2,3);
    v1.print("v1 = ");
    
    std::cout << "\nTest multiplication and division: \n";
    int a = 5;
    vec v4, v5;
    v4 = a * v1;
    v5 = v1 * a;
    v4.print("v4 = 5 * v1 = ");
    v5.print("v5 = v1 * 5 = ");

    std::cout << "\nTest approximate function on vectors: \n";
    std::cout << "approx(v1, v2) = " << (approx(v1,v2, 1e-9, 1e-9) ? "True":"False") << std::endl;
    std::cout << "approx(v4, v5) = " << (approx(v4,v5, 1e-9, 1e-9) ? "True":"False") << std::endl;


    std::cout << "\nTest dot products:\n";
    std::cout << "Dot product within vec class:\n";
    v1.print("v1 = ");
    v2.print("v2 = ");
    std::cout << "v1.dot(v2) = " << v1.dot(v2) << std::endl;
    std::cout << "Dot product outisde vec class:\n";
    std::cout << "dot(v1, v1) = " << dot(v1, v2) << std::endl;

    std::cout << "\nTest vector products:\n";
    vec v6, v7;
    v6 = vec(1, 3, 4);
    v7 = vec(2, 7, -5);
    v6.print("v6 = ");
    v7.print("v7 = ");
    std::cout << "Vector product within vec class:\n";
    std::cout << "v6.cross(v7) = " << v6.cross(v7);
    std::cout << "\nVector product outside vec class:\n";
    std::cout << "cross(v6,v7)" << cross(v6, v7);
    std::cout << "\nExpected result: (-43, 13, 1)";
    return 0;
}
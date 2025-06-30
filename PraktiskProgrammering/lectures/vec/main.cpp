#include<iostream>
#include<vector>
#include<cmath>

using namespace std;

int main(){
    double x=5;
    double y=x; //Copies the value and stores it in a new location
    double& z=x; //Creates a reference to the value of x, thus sharing the location

    x = 10; //Changes the value of x. y will not be affected, but z will be.

    cout << "x = " << x << endl;
    cout << "y = " << y << endl;
    cout << "z = " << z << endl << endl;


    vector<double> a(5); //Creates a vector of size 3 called a
    for(int i=0; i<a.size(); i++){
        a[i] = pow(i,2); //Sets the value of the ith element of a to i
    };
    vector<double> b=a; //Copies the vector a to a new vector b
    vector<double>& c=a; //Creates a reference to the vector a, thus sharing the location

    a[0] = 10; //Changes the value of the first element of a. b will not be affected, but c will be.

    cout << "a = [";
    for(int i=0; i<a.size(); i++){
        cout << a[i] << " ";
    };
    cout << "]" << endl << endl;

    cout << "b = [";
    for(int i=0; i<b.size(); i++){
        cout << b[i] << " ";
    };
    cout << "]" << endl << endl;

    cout << "c = [";
    for(int i=0; i<c.size(); i++){
        cout << c[i] << " ";
    };
    cout << "]" << endl << endl;
}
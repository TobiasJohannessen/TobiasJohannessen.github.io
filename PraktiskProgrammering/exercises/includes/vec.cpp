#include<iostream>
#include<cmath>
#include<string>
#include "vec.h"
#include "sfuns.h"
//#include "sfuns.cpp"

#define SELF (*this)

vec& vec::operator+=(const vec& other){
    x+=other.x; 
    y+=other.y; 
    z+=other.z; 
    return *this;
}

vec& vec::operator-=(const vec& other){
    x-=other.x; 
    y-=other.y; 
    z-=other.z; 
    return *this;
}

vec& vec::operator*=(double c){
    x*=c; 
    y*=c; 
    z*=c; 
    return *this;
}

vec& vec::operator/=(double c){
    x/=c; 
    y/=c; 
    z/=c; 
    return *this;
}

std::ostream& operator<<(std::ostream& os, const vec& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

double vec::dot(const vec& other){
return x*other.x + y*other.y + z*other.z;
}

vec vec::cross(const vec& other) const {
    double new_x, new_y, new_z;
    new_x = y*other.z - z*other.y;
    new_y = z*other.x - x*other.z;
    new_z = x*other.y - y*other.x;
    return vec(new_x, new_y, new_z);
}


vec operator-(const vec& v){return vec(-v.x, -v.y, -v.z);}
vec operator-(const vec& v1, const vec& v2){return vec(v1.x -v2.x, v1.y - v2.y, v1.z - v2.z);}
vec operator+(const vec& v1, const vec& v2){return vec(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);}
vec operator*(const vec& v, double c){return vec(v.x*c, v.y*c, v.z*c);}
vec operator*(double c, const vec& v){ return vec(v.x * c, v.y * c, v.z * c);}
vec operator/(const vec& v, double c){return vec(v.x/c, v.y/c, v.z/c);}

void vec::print(std::string s) const{
    std::cout << s << "(" << x << ", " << y << ", " << z << ")"<< std::endl;
}

// Dot product. Takes two vectors and returns the dot product.
double dot(vec &v, vec &w){
    return v.x*w.x + v.y*w.y + v.z*w.z;
}

vec cross(const vec& v, const vec& w){
    double new_x, new_y, new_z;
    new_x = v.y*w.z - v.z*w.y;
    new_y = v.z*w.x - v.x*w.z;
    new_z = v.x*w.y - v.y*w.x;
    return vec(new_x, new_y, new_z);
}


//Vector approx function. The approx function used on every element is found in sfuns.cpp
bool approx(const vec& u, const vec& v, double acc, double eps){
    if (!::approx(u.x, v.x, acc, eps)) return false;
    if (!::approx(u.y, v.y, acc, eps)) return false;
    if (!::approx(u.z, v.z, acc, eps)) return false;
    return true;
}   
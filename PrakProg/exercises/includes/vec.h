#ifndef HAVE_VEC_H
#define HAVE_VEC_H
#include<iostream>
#include<string>
#include"sfuns.h"

struct vec{
    double x,y,z;
    vec(double x,double y,double z): x(x),y(y),z(z){} // parametrized constructor
    vec():vec(0,0,0){} // default constructor
    vec(const vec&)=default; // copy constructor
    vec(vec&&)=default; // move constructor
    ~vec()=default; // destructor
    vec& operator=(const vec&)=default; // copy assignment
    vec& operator=(vec&&)=default; // move assignment
    vec& operator+=(const vec&);
    vec& operator-=(const vec&);
    vec& operator*=(double);
    vec& operator/=(double c);
    void set(double a,double b,double c){x=a;y=b;z=c;}
    void print(std::string s="") const; // for debugging
    double dot(const vec& other);
    vec cross(const vec& other) const;
    friend std::ostream& operator<<(std::ostream&, const vec&);
};
vec operator-(const vec& v);
vec operator-(const vec& v1, const vec& v2);
vec operator+(const vec& v1, const vec& v2);
vec operator*(const vec& v, double c);
vec operator*(double c, const vec& v);
vec operator/(const vec& v, double c);

double dot(vec& v, vec& w);
vec cross(const vec& v, const vec& w);

bool approx(const vec& u, const vec& v, double acc=1e-9, double eps=1e-9);
#endif
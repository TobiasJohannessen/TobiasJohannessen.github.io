#include"matrix.h"
#include<string>
#include<algorithm>
#include<cmath>
#include<iostream>
#include<cassert>
#include<stdio.h>
#define SELF (*this)
#define FORV(i,v) for(int i=0;i<v.size();i++)
#define FOR_COLS(i,A) for(int i=0;i<A.size2();i++)
namespace pp{

bool approx(NUMBER x,NUMBER y,NUMBER acc=1e-6,NUMBER eps=1e-6){
	if(std::fabs(x-y) < acc)return true;
	if(std::fabs(x-y) < eps*(std::fabs(x)+std::fabs(y)))return true;
	return false;
}

bool approx(const vector& u,const vector& v,NUMBER acc,NUMBER eps){
	if(u.size()!=v.size())return false;
	for(int i=0;i<u.size();i++)if(!approx(u[i],v[i],acc,eps))return false;
	return true;
}

vector& vector::operator+=(const vector& other) {
	FORV(i,SELF) data[i]+=other.data[i];
	return SELF; }

vector& vector::operator-=(const vector& other) {
	FORV(i,SELF) data[i]-=other.data[i];
	return SELF; }

vector& vector::operator*=(NUMBER x) {
	FORV(i,SELF) data[i]*=x;
	return SELF; }

vector& vector::operator/=(NUMBER x) {
	FORV(i,SELF) data[i]/=x;
	return SELF; }

vector& vector::add(NUMBER x){
	data.push_back(x);
	return SELF;}

vector& vector::push_back(NUMBER x){
	data.push_back(x);
	return SELF;}

double vector::norm() const {
	double s=0;
	FORV(i,SELF)s+=SELF[i]*SELF[i];
	return std::sqrt(s);
	};

double vector::dot(const vector& other){
    if (SELF.size() != other.size()){
        std::cout << "ERROR (Dot Product): The two vectors have incompatible lengths." << std::endl;
        return 0;
    };
    double sum = 0;
    for (int i = 0; i<SELF.size(); i++){
        sum += SELF[i] * other[i];
    };
    return sum;
    };

vector vector::map(std::function<double(double)> f) const{
	vector r=SELF;
	for(int i=0;i<r.size();i++)r[i]=f(r[i]);
	return r;
	}

void vector::print(std::string s) const {
	std::cout << s;
	FORV(i,SELF)printf("%9.3g ",(double)SELF[i]);
	printf("\n");
	}

vector operator/(const vector& v, NUMBER x){
	vector r=v;
	r/=x;
	return r; }

vector operator*(const vector& v, NUMBER x){
	vector r=v;
	r*=x;
	return r; }

vector operator*(NUMBER x,const vector& a){ return a*x; }

vector operator+(const vector& a, const vector& b){
	vector r=a;
	r+=b;
	return r; }

vector operator-(const vector& a){
	vector r=a;
	for(int i=0;i<r.size();i++)r[i]=-r[i];
	return r; }

vector operator-(const vector& a, const vector& b){
	vector r=a;
	r-=b;
	return r; }

double dot(const vector& v, const vector& w){
    if (v.size() != w.size()){
        std::cout << "ERROR (Dot Product): The two vectors have incompatible lengths." << std::endl;
        return 0;
    };
    double sum = 0;
    for (int i = 0; i<v.size(); i++){
        sum += v[i] * w[i];
    };
    return sum;
    };





// MATRIX OPERATIONS AND FUNCTIONS

void matrix::resize(int n, int m){
	cols.resize(m);
	for(int i=0;i<m;++i)cols[i].resize(n);
	}

matrix& matrix::operator+=(const matrix& other) {
	FOR_COLS(i,SELF) SELF[i]+=other[i];
	return SELF; }

matrix& matrix::operator-=(const matrix& other) {
	FOR_COLS(i,SELF) SELF[i]-=other[i];
	return SELF; }

matrix& matrix::operator*=(NUMBER x) {
	FOR_COLS(i,SELF) SELF[i]*=x;
	return SELF; }

matrix& matrix::operator/=(NUMBER x) {
	FOR_COLS(i,SELF) SELF[i]/=x;
	return SELF; }

matrix operator/(const matrix& A,NUMBER x){
	matrix R=A;
	R/=x;
	return R; }

matrix operator*(const matrix& A,NUMBER x){
	matrix R=A;
	R*=x;
	return R; }

matrix operator*(NUMBER x,const matrix& A){
	return A*x; }

matrix operator+(const matrix& A, const matrix& B){
	matrix R=A;
	R+=B;
	return R; }

matrix operator-(const matrix& A, const matrix& B){
	matrix R=A;
	R-=B;
	return R; }

vector operator*(const matrix& M, const vector& v){
    if (M.size2() != v.size()){
        std::cout << "ERROR: The number of columns in Matrix M does not equal the number of elements in Vector v!" << std::endl;
        return vector(1);
    };
	vector r; r.resize(M.size1());
	for(int i=0;i<r.size();i++){
		NUMBER sum=0;
		for(int j=0;j<v.size();j++)sum+=M(i,j)*v[j];
		r[i]=sum;
		}
	return r;
	}

matrix operator*(const matrix& A, const matrix& B){
    if (A.size2() != B.size1()){
        std::cout << "ERROR: The number of columns in Matrix A does not equal the number of rows in matrix B" << std::endl;
        return matrix(1,1);
    };
	matrix R(A.size1(),B.size2());
	for(int k=0;k<A.size2();k++)
    {
        for(int j=0;j<B.size2();j++)
        {
            for(int i=0;i<A.size1();i++)R(i,j)+=A(i,k)*B(k,j);
        };
    };    
    return R;
	}

void matrix::setid(){
	assert(size1()==size2());
	for(int i=0;i<size1();i++){
		SELF(i,i)=1;
		for(int j=i+1;j<size1();j++)SELF(i,j)=SELF(j,i)=0;
		}
	}

matrix matrix::transpose() const {
	matrix R(size2(),size1());
	for(int i=0;i<R.size1();i++)
		for(int j=0;j<R.size2();j++) R(i,j)=SELF(j,i);
	return R;
	}

matrix matrix::T() const {return SELF.transpose();}

void matrix::print(std::string s) const {
	std::cout << s << std::endl;
	for(int i=0;i<size1();i++){
		for(int j=0;j<size2();j++)printf("%9.3g ",(double)SELF(i,j));
		printf("\n");
		}
	printf("\n");
	}

// QR decomposition operations:

QR::mtuple QR::decomp(matrix& M){
    matrix Q,R;
    int m = M.size2();
    Q = matrix(M);
    R = matrix(m, m);
    for (int i = 0; i<m; i++){
        R(i,i) = Q[i].norm();
        Q[i]/=R(i,i);
        for (int j = i + 1; j<m; j++){
            R(i,j) = dot(Q[i],Q[j]);
        };
    };
    return std::tuple(Q,R);
};
}//pp
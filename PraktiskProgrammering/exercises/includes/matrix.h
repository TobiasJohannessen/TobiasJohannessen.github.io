#ifndef HAVE_MATRIX_H
#define HAVE_MATRIX_H
#ifdef LONG_DOUBLE
	#define NUMBER long double
#else
	#define NUMBER double
#endif
#include<string>
#include<vector>
#include<initializer_list>
#include<functional>
namespace pp{
struct vector {
	std::vector<NUMBER> data;
	vector(int n) : data(n) {}
	vector(std::initializer_list<double> list) :
		data(list.begin(),list.end()) {}
	vector()			=default;
	vector(const vector&)		=default;
	vector(vector&&)		=default;
	vector& operator=(const vector&)=default;
	vector& operator=(vector&&)	=default;
	int size() const {return data.size();}
	void resize(int n) {data.resize(n);}
	NUMBER& operator[](int i) {return data[i];}
	const NUMBER& operator[](int i) const {return data[i];}
	vector& operator+=(const vector&);
	vector& operator-=(const vector&);
	vector& operator*=(NUMBER);
	vector& operator/=(NUMBER);
	vector& add(NUMBER);
	vector& push_back(NUMBER);
	double norm() const;
	double dot(const vector&);
	void print(std::string s="") const;
	vector map(std::function<double(double)>) const;
};

vector operator+(const vector&, const vector&);
vector operator-(const vector&);
vector operator-(const vector&, const vector&);
vector operator*(const vector&, NUMBER);
vector operator*(NUMBER, const vector&);
vector operator/(const vector&, NUMBER);
bool approx(const vector&, const vector&, NUMBER acc=1e-6, NUMBER eps=1e-6);
double dot(const vector&, const vector&);

struct matrix {
	std::vector<vector> cols;
	matrix()=default;
	matrix(int nrows,int ncols) : cols(ncols, vector(nrows)) {}
	matrix(const matrix& other)=default;
	matrix(matrix&& other)=default;
	matrix& operator=(const matrix& other)=default;
	matrix& operator=(matrix&& other)=default;
	int size1() const {return cols.empty() ? 0 : cols[0].size(); }
	int size2() const {return cols.size();}
	void resize(int n, int m);
	void setid();
	matrix transpose() const;
	matrix T() const;
	
	NUMBER get (int i, int j) {return cols[j][i];}
	void set(int i, int j, NUMBER value){cols[j][i] = value;}
	NUMBER& operator()(int i, int j){return cols[j][i];}
	const NUMBER& operator()(int i, int j) const {return cols[j][i];}
	//NUMBER& operator[](int i, int j){return cols[j][i];}
	//const NUMBER& operator[](int i, int j) const {return cols[j][i];}
	vector& operator[](int i){return cols[i];}
	const vector& operator[](int i) const {return cols[i];}
//	vector get_col(int j);
//	void set_col(int j,vector& cj);

	matrix& operator+=(const matrix&);
	matrix& operator-=(const matrix&);
	matrix& operator*=(const matrix&);
	matrix& operator*=(const NUMBER);
	matrix& operator/=(const NUMBER);
	matrix  operator^(int);

	void print(std::string s="") const;
};

matrix operator+(const matrix&, const matrix&);
matrix operator-(const matrix&, const matrix&);
matrix operator*(const matrix&, const matrix&);
matrix operator*(const matrix&, NUMBER);
matrix operator*(NUMBER, const matrix&);
matrix operator/(const matrix&, NUMBER);
vector operator*(const matrix&, const vector&);


class QR 
{
public: 
	using mtuple = std::tuple<matrix, matrix>; // Type definition
	using vector = std::vector<NUMBER>; // Type definition

	static mtuple decomp(matrix&);
	static vector solve(matrix&, matrix&, vector&);
	static double det(matrix&);
	static matrix inverse(matrix&, matrix&);

	QR() = delete; //Prevent instantiation i.e. this class is only used to call upon functions and not as an object itself.
};
#endif
}



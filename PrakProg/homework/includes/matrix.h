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
#include <stdexcept> // For exceptions
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
	void randomize();
	NUMBER& operator[](int i) {return data[i];}
	const NUMBER& operator[](int i) const {return data[i];}
	vector& operator+=(const vector&);
	vector& operator-=(const vector&);
	vector& operator*=(NUMBER);
	vector& operator/=(NUMBER);
	vector& add(NUMBER);
	vector& push_back(NUMBER);
	double norm() const;
	NUMBER dot(const vector&);
	void print(std::string s="") const;
	vector map(std::function<NUMBER(NUMBER)>) const;
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
	void randomize();
	bool is_upper_triangular();

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
	static matrix identity(int dimension);
	void print(std::string s="") const;
};

matrix operator+(const matrix&, const matrix&);
matrix operator-(const matrix&, const matrix&);
matrix operator*(const matrix&, const matrix&);
matrix operator*(const matrix&, NUMBER);
matrix operator*(NUMBER, const matrix&);
matrix operator/(const matrix&, NUMBER);
vector operator*(const matrix&, const vector&);
matrix symmetric(int dimension); // Create a symmetric matrix with given dimension
bool approx(const matrix& A,const matrix& B,NUMBER acc=1e-6,NUMBER eps=1e-6);

class QR 
{
public: 
	using mtuple = std::tuple<matrix, matrix>; // Type definition
	//using vector = std::vector<NUMBER>; // Type definition

	static mtuple decomp(const matrix&);
	static vector solve(const matrix&, const matrix&, const vector&);
	static vector solve(const matrix&, const vector&);
	static double det(const matrix&);
	static matrix inverse(const matrix&);

	QR() = delete; //Prevent instantiation i.e. this class is only used to call upon functions and not as an object itself.
};

class EVD{
public:
	using mvtuple = std::tuple<matrix, vector>; // Type definition
	using mtuple = std::tuple<matrix, matrix>; // Type definition
	vector w;
	matrix V;
	static void timesJ(matrix& A, int p, int q, NUMBER theta);
	static void Jtimes(matrix& A, int p, int q, NUMBER theta);
	static mtuple cyclic(matrix& A, matrix& V_in);
	EVD(const matrix& M);

};
}
#endif



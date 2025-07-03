
#include"../../includes/matrix.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include<fstream>
#include <vector>

int binsearch(const std::vector<double>& x, double z)
	{/* locates the interval for z by bisection */ 
	if( z<x[0] || z>x[x.size() - 1] ) throw std::runtime_error("binsearch: bad z");
	int i=0, j=x.size()-1;
	while(j-i>1){
		int mid=(i+j)/2;
		if(z>x[mid]) i=mid; else j=mid;
		}
	return i;
	};

class qspline {
public:
    std::vector<double> x, y, b, c;
    qspline(const std::vector<double>& xs, const std::vector<double>& ys){
        if (xs.size() != ys.size()) {
            throw std::runtime_error("qspline: x and y must have the same size");
        }
        if (xs.size() < 2) {
            throw std::runtime_error("qspline: at least two points are required");
        }
        x = xs;
        y = ys;
        b.resize(x.size() - 1);
        c.resize(x.size() - 1);
        std::vector<double> p(x.size() - 1);
        std::vector<double> dx(x.size() - 1);
        c[0] = 0; // Natural spline condition at the first point
        for (size_t i = 0; i < x.size() - 1; ++i) {
            dx[i] = x[i + 1] - x[i];
            if (!(dx[i] > 0)) throw std::runtime_error("qspline: x values must be strictly increasing");
            p[i] = (y[i + 1] - y[i]) / dx[i];
        };
        for (size_t i = 0; i < x.size() - 2; ++i) {
            c[i + 1] = (p[i + 1] - p[i] - c[i] * dx[i]) / dx[i + 1];
        };
        c[x.size() - 2] /= 2; // Natural spline condition at the last point
        for (size_t i = x.size()-3; i < x.size() - 1; --i) {
            c[i] = (p[i + 1] - p[i] - c[i + 1] * dx[i + 1]) / dx[i];
        }
        for (size_t i = 0; i < x.size() - 1; ++i) {
            b[i] = p[i] - c[i] * dx[i];
        }
    }

    double operator()(double z) const {
        int i = binsearch(x, z);
        double dx = z - x[i];
        if (!(dx >= 0 && dx <= x[i + 1] - x[i])) throw std::runtime_error("qspline: z must be in the range of x");
        return y[i] + b[i] * dx + c[i] * dx * dx;
    };
    double integral(double z) const {
        int i = binsearch(x, z);
        double integral = 0.0;
        for (int j = 0; j < i; ++j) {
            double dx = x[j + 1] - x[j];
            if (!(dx > 0)) throw std::runtime_error("qspline: x values must be strictly increasing");
            integral += y[j] * dx + b[j] * dx * dx / 2.0 + c[j] * dx * dx * dx / 3.0;
        }
        double dx_last = z - x[i];
        if (!(dx_last >= 0)) throw std::runtime_error("qspline: z must be >= x[i]");
        //double y_z = (*this)(z);
        integral += y[i] * dx_last + b[i] * dx_last * dx_last / 2.0 + c[i] * dx_last * dx_last * dx_last / 3.0;
        return integral;
    };
    double derivative(double z) const {
        int i = binsearch(x, z);
        double dx = z - x[i];
        if (!(dx >= 0 && dx <= x[i + 1] - x[i])) throw std::runtime_error("qspline: z must be in the range of x");
        return b[i] + 2 * c[i] * dx;
    };

    void print(){
        std::cout << "x: ";
        for (const auto& val : x) std::cout << val << " ";
        std::cout << "\ny: ";
        for (const auto& val : y) std::cout << val << " ";
        std::cout << "\nb: ";
        for (const auto& val : b) std::cout << val << " ";
        std::cout << "\nc: ";
        for (const auto& val : c) std::cout << val << " ";
        std::cout << std::endl;
    };
};


std::vector<double> linspace(double start, double end, int num_points) {
    std::vector<double> result;
    double step = (end - start) / (num_points - 1);  // Calculate step size

    for (int i = 0; i < num_points; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

int main(){
    std::vector<double> xs = {0, 1, 2, 3, 4,5,6,7,8,9,10};
    std::vector<double> ys = std::vector<double>(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        ys[i] = std::sin(xs[i]);
    }

    std::ofstream datafile("quad_spline/data.txt");
    if (!datafile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < xs.size(); ++i) {
        datafile << xs[i] << "\t" << ys[i] << std::endl;
    }
    datafile.close();

    qspline spline(xs, ys);

    


    std::vector<double> xs_spline = linspace(0, 10, 100);
    std::vector<double> ys_spline(xs_spline.size());
    std::vector<double> ys_derivative(xs_spline.size());
    std::vector<double> ys_integral(xs_spline.size());

    std::ofstream splinefile("quad_spline/splines.txt");
    if (!splinefile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < xs_spline.size(); ++i) {
        ys_spline[i] = spline(xs_spline[i]);
        ys_derivative[i] = spline.derivative(xs_spline[i]);
        ys_integral[i] = spline.integral(xs_spline[i]);
        splinefile << xs_spline[i] << "\t" << ys_spline[i] << "\t" << ys_derivative[i] << "\t" << ys_integral[i] << std::endl;
    }
    splinefile.close();

    std::cout << "Data and spline files created successfully." << std::endl;
    return 0;
};
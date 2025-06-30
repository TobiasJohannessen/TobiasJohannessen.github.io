
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

double lin_interp(const std::vector<double>& x, const std::vector<double>& y, double z) {
    int i = binsearch(x, z);
    double dx = x[i+1] - x[i];
    if(!(dx > 0)) throw std::runtime_error("lin_interp: x values must be strictly increasing");
    double dy = y[i+1] - y[i];
    return y[i] + dy/dx * (z - x[i]);
};

double lin_interp_integral(const std::vector<double>& x, const std::vector<double>& y, double z) {
    int i = binsearch(x, z);
    double integral = 0.0;
    for (int j = 0; j < i; ++j) {
        double dx = x[j+1] - x[j];
        if(!(dx > 0)) throw std::runtime_error("lin_interp_integral: x values must be strictly increasing");
        integral += (y[j] + y[j+1])/2.0 * dx; // Trapezoidal rule
    }
    double dx_last = z - x[i];
    if(!(dx_last >= 0)) throw std::runtime_error("lin_interp_integral: z must be >= x[i]");
    double y_z = lin_interp(x, y, z);
    integral += (y[i] + y_z) * dx_last / 2.0;
    return integral;
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
        ys[i] = std::cos(xs[i]);
    }

    std::ofstream datafile("lin_spline/data.txt");
    if (!datafile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < xs.size(); ++i) {
        datafile << xs[i] << "\t" << ys[i] << std::endl;
    }
    datafile.close();


    std::vector<double> xs_spline = linspace(0, 10, 100);
    std::vector<double> ys_spline(xs_spline.size());
    std::vector<double> ys_integral(xs_spline.size());

    std::ofstream splinefile("lin_spline/splines.txt");
    if (!splinefile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < xs_spline.size(); ++i) {
        ys_spline[i] = lin_interp(xs, ys, xs_spline[i]);
        ys_integral[i] = lin_interp_integral(xs, ys, xs_spline[i]);
        splinefile << xs_spline[i] << "\t" << ys_spline[i] << "\t" << ys_integral[i] << std::endl;
    }
    splinefile.close();

    std::cout << "Data and spline files created successfully." << std::endl;
    return 0;
};
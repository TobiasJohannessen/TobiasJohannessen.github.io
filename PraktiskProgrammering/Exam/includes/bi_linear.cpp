#include "bi_linear.h"
#include <tuple>
#include"matrix.h"
#include<vector>
#include<cmath>



//Implement binary search by first doing a binary search on the x vector, then on the y vector. We can then combine the results into a tuple with a 2D-binary search.
int binsearch(const std::vector<double>& x, double z){
    /* locates the interval for z by bisection */ 
	if( z<x[0] || z>x[x.size() - 1] ) throw std::runtime_error("binsearch: bad z");
	int i=0, j=x.size()-1;
	while(j-i>1){
		int mid=(i+j)/2;
		if(z>x[mid]) i=mid; else j=mid;
		}
	return i;
	};


//Methods to generate data sets for a rectilinear grid. The grid is defined by a range in x and y, and the number of points in each direction.

std::vector<double> linspace(double start, double end, int num_points) {
    std::vector<double> result;
    double step = (end - start) / (num_points - 1);  // Calculate step size

    for (int i = 0; i < num_points; ++i) {
        result.push_back(start + i * step);
    }
    return result;
};

std::tuple<std::vector<double>, std::vector<double>, pp::matrix> recti_linear_grid(std::function<double(double, double)> f, std::pair<double, double> x_range, std::pair<double, double> y_range, int n_x, int n_y) {
	// Create a grid of points
	std::vector<double> x_points = linspace(x_range.first, x_range.second, n_x);
	std::vector<double> y_points = linspace(y_range.first, y_range.second, n_y);

	// Create a matrix to hold the values at the grid points
	pp::matrix grid_values(n_y, n_x);
	for (int i = 0; i < n_y; ++i) {
		for (int j = 0; j < n_x; ++j) {
			grid_values(i, j) = f(x_points[j], y_points[i]);
		}
	}


	return {x_points, y_points, grid_values};
};





// Constructor using member initializer list
BiLinearInterpolator::BiLinearInterpolator(const pp::matrix& grid,
						const std::vector<double>& x_points,
						const std::vector<double>& y_points)
	: grid(grid), x_points(x_points), y_points(y_points) {}

// Bilinear interpolation using internal grid and points
double BiLinearInterpolator::interpolate(double x, double y) const {
	int i = binsearch(x_points, x);
	int j = binsearch(y_points, y);

	if (i < 0 || i >= static_cast<int>(x_points.size()) - 1 ||
		j < 0 || j >= static_cast<int>(y_points.size()) - 1) {
		throw std::out_of_range("Coordinates out of bounds for interpolation.");
	}

	double x1 = x_points[i];
	double x2 = x_points[i + 1];
	double y1 = y_points[j];
	double y2 = y_points[j + 1];

	double f11 = grid(j, i);         // f(x1, y1)
	double f12 = grid(j + 1, i);     // f(x1, y2)
	double f21 = grid(j, i + 1);     // f(x2, y1)
	double f22 = grid(j + 1, i + 1); // f(x2, y2)

	double denom = (x2 - x1) * (y2 - y1);
	return (f11 * (x2 - x) * (y2 - y) +
			f21 * (x - x1) * (y2 - y) +
			f12 * (x2 - x) * (y - y1) +
			f22 * (x - x1) * (y - y1)) / denom;
}

// Static method to create an instance of BiLinearInterpolator
BiLinearInterpolator BiLinearInterpolator::create(const pp::matrix& grid,
									const std::vector<double>& x_points,
									const std::vector<double>& y_points) {
	return BiLinearInterpolator(grid, x_points, y_points);
}

// Static method to perform repeated linear interpolation
double BiLinearInterpolator::interpolate(const pp::matrix& grid,
											const std::vector<double>& x_points,
											const std::vector<double>& y_points,
											double x, double y) {
	BiLinearInterpolator interpolator(grid, x_points, y_points);
	return interpolator.interpolate(x, y);
}
pp::matrix BiLinearInterpolator::interpolate_grid(const std::vector<double>& new_x_points,
						const std::vector<double>& new_y_points) const {
							
// Create a new matrix to hold the interpolated values
pp::matrix interpolated_grid(new_y_points.size(), new_x_points.size());
for (size_t i = 0; i < new_y_points.size(); ++i) {
	for (size_t j = 0; j < new_x_points.size(); ++j) {
		// Perform bilinear interpolation for each new point
		interpolated_grid(i, j) = interpolate(new_x_points[j], new_y_points[i]);
	}
}

	return interpolated_grid;
}


// Constructor using member initializer list
BiCubicInterpolator::BiCubicInterpolator(const pp::matrix& grid,
						const std::vector<double>& x_points,
						const std::vector<double>& y_points)
	: grid(grid), x_points(x_points), y_points(y_points) {}

// Bicubic interpolation using internal grid and points
double BiCubicInterpolator::interpolate(double x, double y) const { //Follows the same procedure as https://en.wikipedia.org/wiki/Bicubic_interpolation
	int i = binsearch(x_points, x);
	int j = binsearch(y_points, y);

	if (i < 0 || i >= static_cast<int>(x_points.size()) - 1 ||
		j < 0 || j >= static_cast<int>(y_points.size()) - 1) {
		throw std::out_of_range("Coordinates out of bounds for interpolation.");
	}
	double x1 = x_points[i];
	double x2 = x_points[i + 1];
	double y1 = y_points[j];
	double y2 = y_points[j + 1];
	double delta_x =  x2 - x1;
	double delta_y = y2 - y1;

	pp::vector x_knowns(16); //Build a 16x1 vector to hold function values, derivatives and mixed derivatives on the grid points:
	//Function values at the corners of the grid
	// f(x, y) values
	x_knowns[0] = grid(j, i);         // f(x1, y1)
	x_knowns[1] = grid(j, i + 1);     // f(x2, y1)
	x_knowns[2] = grid(j + 1, i);     // f(x1, y2)
	x_knowns[3] = grid(j + 1, i + 1); // f(x2, y2)

	// f_x at the 4 corners
	x_knowns[4] = (grid(j, i + 1) - grid(j, i)) / delta_x;         // f_x(x1, y1)
	x_knowns[5] = (grid(j + 1, i + 1) - grid(j + 1, i)) / delta_x; // f_x(x1, y2)
	x_knowns[6] = (grid(j, i + 2) - grid(j, i + 1)) / delta_x;  // approx at x2
	x_knowns[7] = (grid(j + 1, i + 2) - grid(j + 1, i + 1)) / delta_x;


	// f_y at the 4 corners
	x_knowns[8]  = (grid(j + 1, i) - grid(j, i)) / delta_y;         // f_y(x1, y1)
	x_knowns[9]  = (grid(j + 1, i + 1) - grid(j, i + 1)) / delta_y; // f_y(x2, y1)
	x_knowns[10] = (grid(j + 1, i) - grid(j, i)) / delta_y;         // f_y(x1, y2) ≈ same as x_knowns[8]
	x_knowns[11] = (grid(j + 1, i + 1) - grid(j, i + 1)) / delta_y; // f_y(x2, y2) ≈ same as x_knowns[9]

	// f_xy at the 4 corners
	x_knowns[12] = (grid(j + 1, i + 1) - grid(j + 1, i) - grid(j, i + 1) + grid(j, i)) / (delta_x * delta_y);       // f_xy at (x1, y1)
	x_knowns[13] = (grid(j + 1, i + 2) - grid(j + 1, i + 1) - grid(j, i + 2) + grid(j, i + 1)) / (delta_x * delta_y); // f_xy at (x2, y1)
	x_knowns[14] = (grid(j + 2, i + 1) - grid(j + 2, i) - grid(j + 1, i + 1) + grid(j + 1, i)) / (delta_x * delta_y); // f_xy at (x1, y2)
	x_knowns[15] = (grid(j + 2, i + 2) - grid(j + 2, i + 1) - grid(j + 1, i + 2) + grid(j + 1, i + 1)) / (delta_x * delta_y); // f_xy at (x2, y2)


	pp::vector alpha(16);

	// Calculate the coefficients for the bicubic interpolation by calculating A^-1 * x_knowns = alpha, where A is given by the following matrix:
	double Ainv[16][16] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
    {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0},
    {-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1},
    {-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1},
    {2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1},
    {4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1}
	};

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			alpha[i] += Ainv[i][j] * x_knowns[j];
		}
	}

	double dx = x - x1/delta_x;
	double dy = y - y1/delta_y;

	double result = 0.0;
	// Calculate the bicubic interpolation using the coefficients
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			result += alpha[i * 4 + j] * std::pow(dx, i) * std::pow(dy, j);
		}
	}
	return result;
}


// Static method to create an instance of BiCubicInterpolator
BiCubicInterpolator BiCubicInterpolator::create(const pp::matrix& grid,
									const std::vector<double>& x_points,
									const std::vector<double>& y_points) {
	return BiCubicInterpolator(grid, x_points, y_points);
}

// Static method to perform bicubic interpolation
double BiCubicInterpolator::interpolate(const pp::matrix& grid,
											const std::vector<double>& x_points,
											const std::vector<double>& y_points,
											double x, double y) {
	BiCubicInterpolator interpolator(grid, x_points, y_points);
	return interpolator.interpolate(x, y);
}
pp::matrix BiCubicInterpolator::interpolate_grid(const std::vector<double>& new_x_points,
						const std::vector<double>& new_y_points) const {
							
// Create a new matrix to hold the interpolated values
pp::matrix interpolated_grid(new_y_points.size(), new_x_points.size());
for (size_t i = 0; i < new_y_points.size(); ++i) {
	for (size_t j = 0; j < new_x_points.size(); ++j) {
		// Perform bicubic interpolation for each new point
		interpolated_grid(i, j) = interpolate(new_x_points[j], new_y_points[i]);
	}
}

	return interpolated_grid;
}



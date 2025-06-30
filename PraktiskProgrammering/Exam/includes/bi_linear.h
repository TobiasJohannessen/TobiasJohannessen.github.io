#ifndef BI_LINEAR_H
#define BI_LINEAR_H

#include "matrix.h"
#include <vector>
#include <tuple>
#include <functional>
#include <utility> // for std::pair

using vec = pp::vector;
using mat = pp::matrix;

// Binary search: returns index i where x[i] <= z < x[i+1]
int binsearch(const std::vector<double>& x, double z);


// Generate linearly spaced points between start and end (num_points total)
std::vector<double> linspace(double start, double end, int num_points);

// Generate rectilinear grid: returns (x_points, y_points, grid matrix) from function f over ranges and resolutions
std::tuple<std::vector<double>, std::vector<double>, pp::matrix> recti_linear_grid(
    std::function<double(double, double)> f,
    std::pair<double, double> x_range,
    std::pair<double, double> y_range,
    int n_x,
    int n_y
);


// Class for bilinear interpolation on a given grid and points
class BiLinearInterpolator {
public:
    pp::matrix grid;
    std::vector<double> x_points;
    std::vector<double> y_points;

    // Constructor
    BiLinearInterpolator(const pp::matrix& grid,
                         const std::vector<double>& x_points,
                         const std::vector<double>& y_points);

    // Bilinear interpolation at (x, y) using internal grid and points
    double interpolate(double x, double y) const;

    // Static factory method to create an interpolator instance
    static BiLinearInterpolator create(const pp::matrix& grid,
                                       const std::vector<double>& x_points,
                                       const std::vector<double>& y_points);

    // Static convenience method: bilinear interpolation at (x, y) with given grid and points
    static double interpolate(const pp::matrix& grid,
                                                const std::vector<double>& x_points,
                                                const std::vector<double>& y_points,
                                                double x, double y);

    // Interpolate the grid at new sets of points (assumes new_x_points and new_y_points are sorted)
    pp::matrix interpolate_grid(const std::vector<double>& new_x_points,
                                const std::vector<double>& new_y_points) const;
};


class BiCubicInterpolator {
public:
    pp::matrix grid;
    std::vector<double> x_points;
    std::vector<double> y_points;

    // Constructor
    BiCubicInterpolator(const pp::matrix& grid,
                        const std::vector<double>& x_points,
                        const std::vector<double>& y_points);

    // Bicubic interpolation at (x, y) using internal grid and points
    double interpolate(double x, double y) const;

    // Static factory method to create an interpolator instance
    static BiCubicInterpolator create(const pp::matrix& grid,
                                      const std::vector<double>& x_points,
                                      const std::vector<double>& y_points);

    // Static convenience method: bicubic interpolation at (x, y) with given grid and points
    static double interpolate(const pp::matrix& grid,
                                                const std::vector<double>& x_points,
                                                const std::vector<double>& y_points,
                                                double x, double y);

    // Interpolate the grid at new sets of points (assumes new_x_points and new_y_points are sorted)
    pp::matrix interpolate_grid(const std::vector<double>& new_x_points,
                                const std::vector<double>& new_y_points) const;
};

#endif // BI_LINEAR_H

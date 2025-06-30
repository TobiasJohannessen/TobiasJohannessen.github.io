#include"includes/bi_linear.h"
#include<iostream>
#include<fstream>
#include<cmath>
#include"includes/matrix.h"



// Define the function to be evaluated on the grid
auto f = [](double x, double y) {
    return std::sin(x) * std::cos(y) + y / 10.0; // Example function
};

// Define the ranges and number of points in each direction from input:
double x_start = 0.0, x_end = 10.0;
double y_start = 0.0, y_end = 12.0;
int n_x = 10, n_y = 10;

// Do the same for the interpolated grid. The ranges are kept the same, but the number of points is increased by a factor of interpolation_resolution:
int interpolation_resolution = 5; // Interpolation resolution factor i.e. how many times the original resolution
int interp_n_x = n_x * interpolation_resolution;
int interp_n_y = n_y * interpolation_resolution;

int main() {
    // Create the rectilinear grid using the function f and specified ranges and resolutions
    std::vector<double> x_points, y_points;
    pp::matrix grid_values;
    std::tie(x_points, y_points, grid_values) = recti_linear_grid(f, 
                                                                 std::make_pair(x_start, x_end), 
                                                                 std::make_pair(y_start, y_end), 
                                                                 n_x, n_y);

    // Output the grid values to a file or console
    std::ofstream output_file("data/grid.txt");
    if (!output_file) {
        std::cerr << "Error: cannot open file " << "data/grid.txt" << std::endl;
        return 1;
    }
    // Clear the file before writing
    output_file.clear();
    output_file << "x\t y \t value\n";
    for (int i = 0; i < n_y; ++i) {
        for (int j = 0; j < n_x; ++j) {
            output_file << x_points[j] << "\t" << y_points[i] << "\t" << grid_values(i, j) << "\n";
        }
        if (i < n_y - 1) {
            output_file << "\n"; // Add a newline between rows
        }
    }
    std::cout << "Data written to " << "data/grid.txt" << std::endl;
    output_file.close();
 
    // Output the values for one constant x-index and one constant y-index for plotting a slice through the grid.
    std::ofstream output_file_x("data/constant_x.txt");
    if (!output_file_x) {
        std::cerr << "Error: cannot open file data/constant_x.txt" << std::endl;
        return 1;
    }
    output_file_x << "x\t y \t value\n";
    // Write the values for one constant x-index:

    int constant_x_index = n_x / 2; // Choose the middle x-index
    for (int i = 0; i < n_y; ++i) {
        output_file_x << x_points[constant_x_index] << "\t" << y_points[i] << "\t" << grid_values(i, constant_x_index) << "\n";
    }
    output_file_x.close();

    std::ofstream output_file_y("data/constant_y.txt");
    if (!output_file_y) {
        std::cerr << "Error: cannot open file constant_y.txt" << std::endl;
        return 1;
    }
    output_file_y << "x\t y \t value\n";
    // Write the values for one constant y-index:
    int constant_y_index = n_y / 2; // Choose the middle y-index
    for (int j = 0; j < n_x; ++j) {
        output_file_y << x_points[j] << "\t" << y_points[constant_y_index] << "\t" << grid_values(constant_y_index, j) << "\n";
    }
    output_file_y.close();

    // Create new x and y points for interpolation 
  
    std::vector<double> new_x_points = linspace(x_start, x_end, interp_n_x);
    std::vector<double> new_y_points = linspace(y_start, y_end, interp_n_y);

    BiLinearInterpolator BLI = BiLinearInterpolator::create(grid_values, x_points, y_points);

    // Interpolate the grid values at the new points
    pp::matrix interpolated_grid = BLI.interpolate_grid(new_x_points, new_y_points);


    // Output the interpolated grid values to a file
    std::ofstream interpolated_file("data/interpolated_grid.txt");
    if (!interpolated_file) {
        std::cerr << "Error: cannot open file data/interpolated_grid.txt" << std::endl;
        return 1;
    }
    interpolated_file << "x\t y \t value\n";
    for (size_t i = 0; i < new_y_points.size(); ++i) {
        for (size_t j = 0; j < new_x_points.size(); ++j) {
            interpolated_file << new_x_points[j] << "\t" << new_y_points[i] << "\t" << interpolated_grid(i, j) << "\n";
        }
        if (i < new_y_points.size() - 1) {
            interpolated_file << "\n"; // Add a newline between rows
        }
    }   
    std::cout << "Interpolated data written to data/interpolated_grid.txt" << std::endl;
    interpolated_file.close();

    //Output the interpolated values for one constant x-index
    std::ofstream interpolated_file_x("data/interpolated_constant_x.txt");
    if (!interpolated_file_x) {
        std::cerr << "Error: cannot open data/file interpolated_constant_x.txt" << std::endl;
        return 1;
    }
    interpolated_file_x << "x\t y \t value\n";
    // Write the values for one constant x-index:
    // Interpolate along same x as original constant_x slice
    double x_val = x_points[constant_x_index];
    for (int i = 0; i < interp_n_y; ++i) {
    double y_val = new_y_points[i];
    double interp_val = BLI.interpolate(x_val, y_val);
    interpolated_file_x << x_val << "\t" << y_val << "\t" << interp_val << "\n";
}
    interpolated_file_x.close();    

    //Output the interpolated values for one constant y-index
    std::ofstream interpolated_file_y("data/interpolated_constant_y.txt");
    if (!interpolated_file_y) {
        std::cerr << "Error: cannot open file data/interpolated_constant_y.txt" << std::endl;
        return 1;
    }
    interpolated_file_y << "x\t y \t value\n";
    // Write the values for one constant y-index:
    // Interpolate along same y as original constant_y slice
    double y_val = y_points[constant_y_index];
    for (int j = 0; j < interp_n_x; ++j) {
        double x_val = new_x_points[j];
        double interp_val = BLI.interpolate(x_val, y_val);
        interpolated_file_y << x_val << "\t" << y_val << "\t" << interp_val << "\n";
    }
    interpolated_file_y.close();

    std::cout << "Interpolated constant x and y data written to files." << std::endl;   
    
     // ------------------BICUBIC INTERPOLATION------------------

    
    // Create new x and y points for bicubic interpolation
    // These points are offset by 2 units from the original grid to avoid edge effects
    new_x_points = linspace(x_start + 2, x_end - 2, interp_n_x);
    new_y_points = linspace(y_start + 2, y_end - 2, interp_n_y);    


    // Create the bicubic interpolator
    BiCubicInterpolator BCI(grid_values, x_points, y_points);
    // Interpolate the grid values at the new points using bicubic interpolation
    pp::matrix bicubic_interpolated_grid = BCI.interpolate_grid(new_x_points, new_y_points);

    // Output the bicubic interpolated grid values to a file
    std::ofstream bicubic_interpolated_file("data/bicubic_interpolated_grid.txt");
    if (!bicubic_interpolated_file) {
        std::cerr << "Error: cannot open file data/bicubic_interpolated_grid.txt" << std::endl;
        return 1;
    }
    bicubic_interpolated_file << "x\t y \t value\n";
    for (size_t i = 0; i < new_y_points.size(); ++i) {
        for (size_t j = 0; j < new_x_points.size(); ++j) {
            bicubic_interpolated_file << new_x_points[j] << "\t" << new_y_points[i] << "\t" 
                                      << bicubic_interpolated_grid(i, j) << "\n";
        }
        if (i < new_y_points.size() - 1) {
            bicubic_interpolated_file << "\n"; // Add a newline between rows
        }
    }
    std::cout << "Bicubic interpolated data written to data/bicubic_interpolated_grid.txt" << std::endl;
    bicubic_interpolated_file.close();

    // Output the bicubic interpolated values for one constant x-index
    std::ofstream bicubic_interpolated_file_x("data/bicubic_interpolated_constant_x.txt");
    if (!bicubic_interpolated_file_x) {
        std::cerr << "Error: cannot open file data/bicubic_interpolated_constant_x.txt" << std::endl;
        return 1;
    }
    bicubic_interpolated_file_x << "x\t y \t value\n";
    // Write the values for one constant x-index:
    // Interpolate along same x as original constant_x slice
    double bicubic_x_val = x_points[constant_x_index];

    for (int i = 0; i < interp_n_y; ++i) {
        double bicubic_y_val = new_y_points[i];
        double bicubic_interp_val = BCI.interpolate(bicubic_x_val, bicubic_y_val);
        bicubic_interpolated_file_x << bicubic_x_val << "\t" << bicubic_y_val << "\t" 
                                    << bicubic_interp_val << "\n";
    }
    bicubic_interpolated_file_x.close();
    // Output the bicubic interpolated values for one constant y-index
    std::ofstream bicubic_interpolated_file_y("data/bicubic_interpolated_constant_y.txt");
    if (!bicubic_interpolated_file_y) {
        std::cerr << "Error: cannot open file data/bicubic_interpolated_constant_y.txt" << std::endl;
        return 1;
    }
    bicubic_interpolated_file_y << "x\t y \t value\n";
    // Write the values for one constant y-index:
    // Interpolate along same y as original constant_y slice
    double bicubic_y_val = y_points[constant_y_index];
    for (int j = 0; j < interp_n_x; ++j) {
        double bicubic_x_val = new_x_points[j];
        double bicubic_interp_val = BCI.interpolate(bicubic_x_val, bicubic_y_val);
        bicubic_interpolated_file_y << bicubic_x_val << "\t" << bicubic_y_val << "\t" 
                                    << bicubic_interp_val << "\n";
    }
    bicubic_interpolated_file_y.close();
    std::cout << "Bicubic interpolated constant x and y data written to files." << std::endl;

    std::ofstream bilinear_diff("data/bilinear_diffs.txt");
    if (!bilinear_diff) {
        std::cerr << "Error: cannot open file data/bilinear_diff.txt" << std::endl;
        return 1;
    }
    bilinear_diff << "x\t y \t diff\n";
    for (size_t i = 0; i < new_y_points.size(); ++i) {
        for (size_t j = 0; j < new_x_points.size(); ++j) {
            double bilinear_val = BLI.interpolate(new_x_points[j], new_y_points[i]);
            bilinear_diff << new_x_points[j] << "\t" << new_y_points[i] << "\t"
                          << std::abs(bilinear_val - f(new_x_points[j], new_y_points[i])) << "\n";
            }
        if (i < new_y_points.size() - 1) {
            bilinear_diff << "\n"; // Add a newline between rows
        }
    }
    std::cout << "Bilinear interpolation differences written to data/bilinear_diffs.txt" << std::endl;
    bilinear_diff.close(); 

    std::ofstream bicubic_diff("data/bicubic_diffs.txt");
    if (!bicubic_diff) {
        std::cerr << "Error: cannot open file data/bicubic_diffs.txt" << std::endl;
        return 1;
    }
    bicubic_diff << "x\t y \t diff\n";
    for (size_t i = 0; i < new_y_points.size(); ++i) {
        for (size_t j = 0; j < new_x_points.size(); ++j) {
            double bicubic_val = BCI.interpolate(new_x_points[j], new_y_points[i]);
            bicubic_diff << new_x_points[j] << "\t" << new_y_points[i] << "\t"
                         << std::abs(bicubic_val - f(new_x_points[j], new_y_points[i])) << "\n";
        }
        if (i < new_y_points.size() - 1) {
            bicubic_diff << "\n"; // Add a newline between rows
        }
    }
    std::cout << "Bicubic interpolation differences written to data/bicubic_diffs.txt" << std::endl;
    bicubic_diff.close();
  
     
    return 0;
}

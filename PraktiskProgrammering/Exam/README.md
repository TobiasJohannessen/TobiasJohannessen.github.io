I have done the examination project <strong>Bi-linear interpolation on a rectilinear grid in two dimensions</strong> given by the description:

"""
<strong>Introduction</strong>

A rectilinear grid (note that rectilinear is not necessarily cartesian nor regular) in two dimensions is a set of nx×ny points where each point can be adressed by a double index (i,j) where 1 ≤ i ≤ nx, 1 ≤ j ≤ ny and the coordinates of the point (i,j) are given as (xi,yj), where x and y are vectors with sizes nx and ny correspondingly. The values of the tabulated function F at the grid points can then be arranged as a matrix {Fi,j=F(xi,yj)}.

<strong>Problem</strong>

Build an interpolating routine which takes as the input the vectors {xi} and {yj}, and the matrix {Fi,j} and returns the bi-linear interpolated value of the function at a given 2D-point p=(px,py).

<strong>Hints</strong>
See the chapter "Bi-linear interpolation" in the book.

The signature of the interpolating subroutine can be

static double bilinear(double[] x, double[] y, matrix F, double px, double py)<br>

"""


<strong>Implementation</strong>

I've done this by implementing different functions in the file bi_linear.cpp. First to make a grid, I've defined linspace, which works as np.linspace in python. I've then made the function recti_linear_grid, which takes x_range, y_range and number of points as well as a function f(x,y) to output two vectors x,y and a matrix Z, which holds the function value at each point.

Then came the work on the interpolator. I used my matrix class written in an earlier Homework question and implemented the Repeated Linear Interpolation method from the Wikipedia page (<a href="https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation">link</a>), which works by interpolating in the x-direction with a constant y and then in the y-direction for a constant x. This can be written neatly as product: vector_1^T * Matrix * vector_2, where both the matrix and the vectors are easy to find.

I placed the Interpolation routine inside a class BiLinearInterpolator, which takes the original grid as an input and saves them. It has a function to perform interpolation in a single point and a function to do it for all points in a grid.

<strong>main.cpp</strong>:

The main.cpp-file is built in the way, that you define everything you need in the beginning of the script i.e. the ranges of x, y, number of points in each direction and the function f(x,y) to be evaluated on the grid. You also choose how many more points, the interpolation grid should have (by multiplying a factor).

The script then pretty much handles the rest. It saves the original grid as well as slices through at a constant x-value and constant y-value. It then performs the interpolation using the functions above and saves them as well. All the output-files can be found in the data-folder (after making).

<strong>Makefile</strong>:

The Makefile is built in the way that it recompiles the includes-files if necessary, compiles and runs the main-file and produces plots from the data-files. Everything happens automatically, if you write make. 

The plots can be found in the plots-folder, where the gnuplot-files to generate each plot is found in plots/plot_functions. 

I've currently made 2 types of plots: A zoom in on the grid, and a multiplot with the grid on top and slices through some x- and y-value below. I've made those plots for the original grid and with interpolation, where the original points are visible for comparison.

<strong>Extra Possibilities</strong>

If time permits, it would be nice to implement the bicubic interpolation method or the polynomial version of the interpolation method and check whether it provides the same interpolation or not.

Update: I've tried to implement the bicubic interpolation, but it got too complicated that I didn't want to spend too much time on it. You can see the best implementation I got in plots/bicubic_*.png.

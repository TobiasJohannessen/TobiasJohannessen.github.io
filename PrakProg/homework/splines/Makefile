CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

LS = lin_spline

.PHONY: all clean

all: lin_spline/main quad_spline/main
	./lin_spline/main
	./quad_spline/main

plot.all: plot.lin.spline plot.quad.spline


lin_spline/main: lin_spline/main.o $(INCLUDE_DIR)/matrix.o 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^


lin_spline/main.o: lin_spline/main.cpp $(INCLUDE_DIR)/matrix.h 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@


plot.lin.spline: lin_spline/data.txt lin_spline/splines.txt
	rm -f plots/linear_spline.svg
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/linear_spline.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"lin_spline/data.txt" using 1:2 with points title "Original Data", \
			"lin_spline/splines.txt" using 1:2 with lines title "Linear Spline", \
			"lin_spline/splines.txt" using 1:3 with lines title "Linear Spline Anti-Derivative"; \
	' | gnuplot


quad_spline/main: quad_spline/main.o $(INCLUDE_DIR)/matrix.o 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^


quad_spline/main.o: quad_spline/main.cpp $(INCLUDE_DIR)/matrix.h 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@


plot.quad.spline: quad_spline/data.txt quad_spline/splines.txt
	rm -f plots/quadratic_spline.svg
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/quadratic_spline.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"quad_spline/data.txt" using 1:2 with points title "Original Data", \
			"quad_spline/splines.txt" using 1:2 with lines title "Quadratic Spline", \
			"quad_spline/splines.txt" using 1:3 with lines title "Quadratic Spline Derivative", \
			"quad_spline/splines.txt" using 1:4 with lines title "Quadratic Spline Anti-Derivative"; \
	' | gnuplot

# Clean up compiled objects and executables
clean:
	rm -f *.txt */*.txt */main */*.o

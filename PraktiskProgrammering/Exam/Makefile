CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out run_test_timing

all: main plot.all
	

# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o $(INCLUDE_DIR)/matrix.o $(INCLUDE_DIR)/bi_linear.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main

main.o: main.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(INCLUDE_DIR)/bi_linear.o: $(INCLUDE_DIR)/bi_linear.cpp $(INCLUDE_DIR)/bi_linear.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@



plot.grid: 
	gnuplot plots/plot_functions/plot_grid.gp

plot.grid.slices:
	gnuplot plots/plot_functions/plot_slices.gp

plot.interp.grid:
	gnuplot plots/plot_functions/plot_interpolated_grid.gp

plot.interp.slices:
	gnuplot plots/plot_functions/plot_interpolated_slices.gp

plot.bicubic.interp.grid:
	gnuplot plots/plot_functions/plot_bicubic_interpolated_grid.gp

plot.bicubic.interp.slices:
	gnuplot plots/plot_functions/plot_bicubic_interpolated_slices.gp

plot.compare:
	gnuplot plots/plot_functions/plot_compare.gp


plot.all: plot.grid plot.grid.slices plot.interp.grid plot.interp.slices plot.bicubic.interp.grid plot.bicubic.interp.slices
	

clean:
	rm -f *.o  data/*.txt $(INCLUDE_DIR)/*.o plots/*.png main

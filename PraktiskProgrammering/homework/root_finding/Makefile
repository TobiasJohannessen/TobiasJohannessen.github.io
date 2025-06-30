CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out 

all: main plot.wavefunctions plot.convergences

# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o $(INCLUDE_DIR)/matrix.o $(INCLUDE_DIR)/ODE.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main > out.txt

main.o: main.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Rule to build the 'ODE.o' object from the matrix implementation file
$(INCLUDE_DIR)/ODE.o: $(INCLUDE_DIR)/ODE.cpp $(INCLUDE_DIR)/ODE.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@


plot.wavefunctions:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/hydrogen_wf.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/hydrogen_wf.txt" using 1:2 with lines title "Numerical Solution" , \
			"data/hydrogen_wf.txt" using 1:3 with lines title "Exact Solution"; \
	' | gnuplot

plot.convergences:
	echo ' \
		set terminal svg size 1000,1000 background rgb "white"; \
		set output "plots/convergence.svg"; \
		set multiplot layout 2,2 title "Convergence Tests" margins 0.13,0.9,0.1,0.9 spacing 0.08,0.08; \
		\
		unset logscale; \
		set key left; set xlabel "r_{min}"; set ylabel "Error"; \
		plot "data/convergence_rmin.txt" using 1:2 with lines title "Varying r_{min}"; \
		\
		unset logscale; \
		set key left; set xlabel "r_{max}"; set ylabel "Error"; \
		plot "data/convergence_rmax.txt" using 1:2 with lines title "Varying r_{max}"; \
		\
		set logscale x; \
		set format x "10^{%T}"; \
		set key left; set xlabel "acc"; set ylabel "Error"; \
		plot "data/convergence_acc.txt" using 1:2 with lines title "Varying acc"; \
		\
		set logscale x; \
		set format x "10^{%T}"; \
		set format y "10^{%T}"; \
		set key left; set xlabel "eps"; set ylabel "Error"; \
		plot "data/convergence_eps.txt" using 1:2 with lines title "Varying eps"; \
		unset multiplot; \
	' | gnuplot


clean:
	rm -f *.o $(INCLUDE_DIR)/*.o main out.txt

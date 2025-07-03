CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out run_test_timing

all: main plot.all plot.orbit.all



# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o $(INCLUDE_DIR)/matrix.o $(INCLUDE_DIR)/ODE.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main

main.o: main.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(INCLUDE_DIR)/ODE.o: $(INCLUDE_DIR)/ODE.cpp $(INCLUDE_DIR)/ODE.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

plot.all: plot.exponential plot.harmonic plot.dampened_oscillator plot.lotka_volterra plot.together

plot.exponential:
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/functions/exponential.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/exponential.txt" using 1:2 with lines title "Exponential", \
	' | gnuplot

plot.harmonic: 
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/functions/harmonic.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/harmonic.txt" using 1:2 with lines title "Harmonic", \
	' | gnuplot

plot.dampened_oscillator:
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/functions/dampened_oscillator.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/dampened_oscillator.txt" using 1:2 with lines title "Dampened Oscillator", \
	' | gnuplot


plot.lotka_volterra:
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/functions/lotka_volterra.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/lotka_volterra.txt" using 1:2 with lines title "Prey", \
			"data/lotka_volterra.txt" using 1:3 with lines title "Predator", \
	' | gnuplot

plot.together:
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/functions/together.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		set yrange [-5:25]; \
		plot  \
			"data/exponential.txt" using 1:2 with lines title "Exponential", \
			"data/harmonic.txt" using 1:2 with lines title "Harmonic", \
			"data/dampened_oscillator.txt" using 1:2 with lines title "Dampened Oscillator", \
	' | gnuplot

plot.orbit.all: plot.orbit.orbits plot.orbit.circular plot.orbit.elliptical plot.orbit.relativistic


plot.orbit.orbits: 
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/orbits/orbit_all.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/circular_orbit.txt" using (1/$$2)*cos($$1):(1/$$2)*sin($$1) with lines title "Circular Orbit", \
			"data/elliptical_orbit.txt" using (1/$$2)*cos($$1):(1/$$2)*sin($$1) with lines title "Elliptical Orbit", \
			"data/relativistic_orbit.txt" using (1/$$2)*cos($$1):(1/$$2)*sin($$1) with lines title "Relativistic Orbit", \
	' | gnuplot

plot.orbit.circular:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/orbits/circular_orbit.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/circular_orbit.txt" using (1/$$2)*cos($$1):(1/$$2)*sin($$1) with lines title "Circular Orbit", \
	' | gnuplot

plot.orbit.elliptical:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/orbits/elliptical_orbit.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/elliptical_orbit.txt" using (1/$$2)*cos($$1):(1/$$2)*sin($$1) with lines title "Elliptical Orbit", \
	' | gnuplot

plot.orbit.relativistic:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/orbits/relativistic_orbit.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/relativistic_orbit.txt" using (1/$$2)*cos($$1):(1/$$2)*sin($$1) with lines title "Relativistic Orbit", \
	' | gnuplot






clean:
	rm -f *.o main *.txt $(INCLUDE_DIR)/*.o plots/* plots/*/*.svg

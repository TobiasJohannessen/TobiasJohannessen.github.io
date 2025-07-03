CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out 

all: main plot.all
# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main > out.txt


main.o: main.cpp 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<


plot.all: plot.errors plot.halton plot.lattice

plot.errors: plot.errors.circle plot.errors.halton plot.errors.lattice plot.errors.compare

plot.errors.circle:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Monte Carlo Circle Area Estimation Errors (Logarithmic)"; \
		set output "plots/circle_errors, log.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set logscale x; \
		set logscale y; \
		set xrange [90:1000000]; \
		f(x) = a/x**0.5; \
		fit f(x) "data/circle_errors.txt" using 1:3 via a; \
		plot  \
			"data/circle_errors.txt" using 1:3 with lines title "Error", \
			f(x) with lines title sprintf("Fitted O(1/\sqrt(N)) (a=%.2e)", a); \
	' | gnuplot
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/circle_errors.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set xrange [0:1000000]; \
		f(x) = a/x**0.5; \
		fit f(x) "data/circle_errors.txt" using 1:3 via a; \
		plot  \
			"data/circle_errors.txt" using 1:3 with lines title "Error", \
			f(x) with lines title sprintf("Fitted O(1/sqrt(N)) (a=%.2e)", a); \
	' | gnuplot

plot.errors.halton:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Halton Sequence Errors (Logarithmic)"; \
		set output "plots/halton_errors, log.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set logscale x; \
		set logscale y; \
		set xrange [90:1000000]; \
		f(x) = a/x**0.5; \
		fit f(x) "data/halton_errors.txt" using 1:3 via a; \
		plot  \
			"data/halton_errors.txt" using 1:3 with lines title "Error", \
			f(x) with lines title sprintf("Fitted O(1/\sqrt(N)) (a=%.2e)", a); \
	' | gnuplot
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/halton_errors.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set xrange [0:1000000]; \
		f(x) = a/x**0.5; \
		fit f(x) "data/halton_errors.txt" using 1:3 via a; \
		plot  \
			"data/halton_errors.txt" using 1:3 with lines title "Error", \
			f(x) with lines title sprintf("Fitted O(1/sqrt(N)) (a=%.2e)", a); \
	' | gnuplot

plot.errors.lattice:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Lattice Point Errors (Logarithmic)"; \
		set output "plots/lattice_errors, log.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set logscale x; \
		set logscale y; \
		set xrange [90:1000000]; \
		f(x) = a/x**0.5; \
		fit f(x) "data/lattice_errors.txt" using 1:3 via a; \
		plot  \
			"data/lattice_errors.txt" using 1:3 with lines title "Error", \
			f(x) with lines title sprintf("Fitted O(1/\sqrt(N)) (a=%.2e)", a); \
	' | gnuplot
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/lattice_errors.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set xrange [0:1000000]; \
		f(x) = a/x**0.5; \
		fit f(x) "data/lattice_errors.txt" using 1:3 via a; \
		plot  \
			"data/lattice_errors.txt" using 1:3 with lines title "Error", \
			f(x) with lines title sprintf("Fitted O(1/sqrt(N)) (a=%.2e)", a); \
	' | gnuplot

plot.errors.compare:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Monte Carlo Circle Area Estimation Errors Comparison (Logarithmic)"; \
		set output "plots/compare_errors, log.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set logscale x; \
		set logscale y; \
		set xrange [90:1000000]; \
		f(x) = a/x**0.5; \
		g(x) = b/x**0.5; \
		h(x) = c/x**0.5; \
		fit f(x) "data/circle_errors.txt" using 1:3 via a; \
		fit g(x) "data/halton_errors.txt" using 1:3 via b; \
		fit h(x) "data/lattice_errors.txt" using 1:3 via c; \
		plot  \
			"data/circle_errors.txt" using 1:3 with lines title "Circle Error", \
			f(x) with lines title sprintf("Circle Fitted O(1/\sqrt(N)) (a=%.2e)", a), \
			"data/halton_errors.txt" using 1:3 with lines title "Halton Error", \
			g(x) with lines title sprintf("Halton Fitted O(1/\sqrt(N)) (b=%.2e)", b), \
			"data/lattice_errors.txt" using 1:3 with lines title "Lattice Error", \
			h(x) with lines title sprintf("Lattice Fitted O(1/\sqrt(N)) (c=%.2e)", c); \
	' | gnuplot
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/compare_errors.svg"; \
		set key left ;\
		set xlabel "Sample Size";\
		set ylabel "Error";\
		set xrange [0:1000000]; \
		f(x) = a/x**0.5; \
		g(x) = b/x**0.5; \
		h(x) = c/x**0.5; \
		fit f(x) "data/circle_errors.txt" using 1:3 via a; \
		fit g(x) "data/halton_errors.txt" using 1:3 via b; \
		fit h(x) "data/lattice_errors.txt" using 1:3 via c; \
		plot  \
			"data/circle_errors.txt" using 1:3 with lines title "Circle Error", \
			f(x) with lines title sprintf("Circle Fitted O(1/sqrt(N)) (a=%.2e)", a), \
			"data/halton_errors.txt" using 1:3 with lines title "Halton Error", \
			g(x) with lines title sprintf("Halton Fitted O(1/sqrt(N)) (b=%.2e)", b), \
			"data/lattice_errors.txt" using 1:3 with lines title "Lattice Error", \
			h(x) with lines title sprintf("Lattice Fitted O(1/sqrt(N)) (c=%.2e)", c); \
	' | gnuplot


plot.halton:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Halton Sequence"; \
		set output "plots/halton_sequence.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/halton_sequence.txt" using 1:2 with points pt 7 ps 0.5 notitle, \
	' | gnuplot

plot.lattice:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Lattice Points"; \
		set output "plots/additive_lattice.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y";\
		plot  \
			"data/additive_lattice.txt" using 1:2 with points pt 7 ps 0.5 notitle, \
	' | gnuplot
clean:
	rm -f *.o main data/*.txt $(INCLUDE_DIR)/*.o plots/* plots/*/*.svg

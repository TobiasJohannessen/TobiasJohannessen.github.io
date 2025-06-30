CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++17 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out 
all: main plot.all test.python


test.python: 
	python3 integrate.py


# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o $(INCLUDE_DIR)/matrix.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main > out.txt


main.o: main.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@


plot.all: plot.erf plot.acc

plot.erf:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/error_function.svg"; \
		set key left ;\
		set xlabel "z";\
		set ylabel "erf(z)";\
		set yrange [-0.01:1.1]; \
		plot  \
			"data/erf_results.txt" using 1:2 with lines title "Computed error function", \
			"data/erf_results.txt" using 1:3 with lines title "Exact error function"; \
	' | gnuplot

plot.acc:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/accuracy.svg"; \
		set key left ;\
		set xlabel "Accuracy";\
		set ylabel "Absolute Error";\
		set logscale x; \
		set logscale y; \
		plot  \
			"data/erf_accuracy.txt" using 1:2 with points pointtype 2 notitle", \
	' | gnuplot


clean:
	rm -f *.o main *.txt $(INCLUDE_DIR)/*.o plots/* plots/*/*.svg

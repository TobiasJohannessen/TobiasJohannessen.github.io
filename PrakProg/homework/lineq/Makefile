CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out run_test_timing

all: main test_executable

# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o matrix.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main > out.txt

main.o: main.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Rule to build the 'test_executable' which will be used for timing
test_executable: test.o matrix.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^

test.o: test.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Target to run the main executable (e.g., for general program execution)
out: main
	./main > out.txt

# Target to generate the timing data
out.times.data: test_executable
	>$@
	# Loop through N values, run the test_executable, and append timing data
	for N in $$(seq 200 20 500); do \
		time --format "$$N %e" --output $@ --append \
		./$< -size:$$N 1>out 2>err ;\
	done

plot.times: out.times.data Makefile
	@echo "Generating plot and fitting N^3 curve..."
	@echo 'set terminal pngcairo enhanced font "Arial,12"' > log.times.gpi
	@echo 'set output "qr_time_plot.png"' >> log.times.gpi
	@echo 'set title "QR Factorization Time vs. Matrix Size (N), (Fit: O(N^3) Complexity)"' >> log.times.gpi
	@echo 'set xlabel "Matrix Size N"' >> log.times.gpi
	@echo 'set ylabel "Time (seconds)"' >> log.times.gpi
	@echo 'set grid' >> log.times.gpi
	@echo 'f(x) = a * x**3' >> log.times.gpi
	@echo "fit f(x) 'out.times.data' using 1:2 via a" >> log.times.gpi
	@echo "plot 'out.times.data' using 1:2 with points pt 7 lc rgb 'blue' title 'Measured Data', \\" >> log.times.gpi
	@echo "     f(x) with lines lc rgb 'red' lw 2 title sprintf('Fitted O(N^3) (a=%.2e)', a)" >> log.times.gpi
	@echo "print sprintf('Fitted coefficient a for O(N^3): %.4e', a)" >> log.times.gpi
	gnuplot log.times.gpi
	@echo "Plot saved to qr_time_plot.png"

# Clean up compiled objects and executables
clean:
	rm -f *.o main test_executable out out.txt err out.times.data

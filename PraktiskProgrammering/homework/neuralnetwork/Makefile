CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out 

all: main plot.all

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

plot.all: plot.network plot.network.derivatives

plot.network:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/neural_network.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y(x)"; \
		plot "data/ann_output_untrained.txt" using 1:2 with lines title "Untrained Model", \
		     "data/ann_output_trained.txt" using 1:2 with lines title "Trained Model", \
			 "data/output_exact.txt" using 1:2 with lines title "Exact Solution", \
			 "data/training_data.txt" using 1:2 with points pointtype 7 pointsize 0.5 title "Training Data"; \
	' | gnuplot

plot.network.derivatives:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/neural_network_derivatives.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "y(x)"; \
		plot "data/ann_output_untrained.txt" using 1:2 with lines title "Untrained Model", \
		     "data/ann_output_trained.txt" using 1:2 with lines title "Trained Model", \
			 "data/ann_output_trained.txt" using 1:3 with lines ls 6 title "1st derivative", \
			 "data/ann_output_trained.txt" using 1:4 with lines title "2nd derivative", \
			 "data/ann_output_trained.txt" using 1:5 with lines title "Anti-derivative",\
			 "data/output_exact.txt" using 1:2 with lines title "Exact Solution", \
			 "data/training_data.txt" using 1:2 with points pointtype 7 pointsize 0.5 title "Training Data"; \
	' | gnuplot



clean:
	rm -f *.o $(INCLUDE_DIR)/*.o main out.txt data/*.txt plots/*.svg

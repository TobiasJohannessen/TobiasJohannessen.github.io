CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean

all: main higgs plot.higgs

# Rule to build the 'main' executable (for general functionality testing, if any)
main: main.o $(INCLUDE_DIR)/matrix.o $(INCLUDE_DIR)/minimization.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main > out.txt

main.o: main.cpp $(INCLUDE_DIR)/matrix.h $(INCLUDE_DIR)/minimization.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

higgs: higgs.o $(INCLUDE_DIR)/matrix.o $(INCLUDE_DIR)/minimization.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./higgs > out_higgs.txt

higgs.o: higgs.cpp $(INCLUDE_DIR)/matrix.h $(INCLUDE_DIR)/minimization.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(INCLUDE_DIR)/minimization.o: $(INCLUDE_DIR)/minimization.cpp $(INCLUDE_DIR)/minimization.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

plot.higgs:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set output "plots/higgs_plot.svg"; \
		set key left ;\
		set ylabel "σ(E)";\
		set xlabel "Energy (GeV)";\
		plot  \
			"data/higgs_fit.txt" using 1:2 with lines title "Fit", \
			"data/higgs.txt" using 1:2:3 with yerrorbars title "Data"; \
	' | gnuplot



clean:
	rm -f *.o $(INCLUDE_DIR)/*.o main out.txt

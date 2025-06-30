CXX = g++
# -O3 for optimization in timing measurements
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
# Define the path to your include files
INCLUDE_DIR = ../includes
# Add the include directory to the compiler flags
CPPFLAGS = -I$(INCLUDE_DIR)

.PHONY: all clean out test hydrogen


all: hydrogen test
	


hydrogen: hydrogen.exe
	./hydrogen.exe -rmax 10.0 -dr 0.1

hydrogen.exe: hydrogen.o matrix.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^

hydrogen.o: hydrogen.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

test: test.exe
	./test.exe

test.exe: test.o matrix.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^


test.o: test.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@


out: test.exe hydrogen.exe
	./test.exe > test_output.txt
	./hydrogen.exe > hydrogen_output.txt

convergence: hydrogen.exe
	rm -f dr_e0.txt rmax_e0.txt # Clean up previous run files
	# Loop for rmax convergence
	for i in $(shell seq 0.2 0.1 10.0); do \
		./hydrogen.exe -rmax $$i -dr 0.1 -conv rmax; \
	done
	# Loop for dr convergence
	for i in $(shell seq 0.01 0.01 1.0); do \
		./hydrogen.exe -rmax 10.0 -dr $$i -conv dr; \
	done

wavefunction: hydrogen.exe
	rm -f wavefunction.txt # Clean up previous wavefunction file
	./hydrogen.exe -rmax 10.0 -dr 0.1 -conv wavefunction

plot.convergence.rmax: rmax_e0.txt
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/rmax_convergence.svg"; \
		set key left ;\
		set xlabel "r_{max} (Bohr Radius)";\
		set ylabel "Ground State Energy (Hartree)";\
		plot  \
			"rmax_e0.txt" index 0 with lines title "r_{max}" \
	'  | gnuplot

plot.convergence.dr: dr_e0.txt
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/dr_convergence.svg"; \
		set key left ;\
		set xlabel "dr (Bohr Radius)";\
		set ylabel "Ground State Energy (Hartree)";\
		plot  \
			"dr_e0.txt" index 0 with lines title "dr" \
	' | gnuplot

plot.wavefunction: wavefunction.txt
	echo ' \
		set terminal svg background rgb "white"; \
		set output "plots/wavefunction.svg"; \
		set key inside right top; \
		set xlabel "r (Bohr Radius)"; \
		set ylabel "Wavefunction (a.u.)"; \
		set title "Lowest s-wave states of Hydrogen Atom"; \
		plot \
			"wavefunction.txt" using 1:2 with lines title "n=1 (Ground State)", \
			"wavefunction.txt" using 1:3 with lines title "n=2 (First Excited State)", \
			"wavefunction.txt" using 1:4 with lines title "n=3 (Second Excited State)", \
			"wavefunction.txt" using 1:5 with lines title "n=4 (Third Excited State)" \
	' | gnuplot

plot.all: plot.convergence.rmax plot.convergence.dr plot.wavefunction
	echo "Plots generated successfully."

clean:
	rm -f *.o *.exe main *.txt

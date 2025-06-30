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
main: main.o $(INCLUDE_DIR)/matrix.o
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^
	./main > out.txt

main.o: main.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<

# Rule to build the 'matrix.o' object from the matrix implementation file
$(INCLUDE_DIR)/matrix.o: $(INCLUDE_DIR)/matrix.cpp $(INCLUDE_DIR)/matrix.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

plot.all: plot.decay.log plot.decay


plot.decay.log: plot.decay.log.sigma_A plot.decay.log.sigma_lambda
plot.decay: plot.decay.sigma_A plot.decay.sigma_lambda

plot.decay.log.sigma_A:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Decay Law Fit, ln(y) = ln(a) - λt"; \
		set output "plots/decay_log_fit_sigmaA.svg"; \
		set key left ;\
		set xlabel "t";\
		set ylabel "log(y)";\
		plot  \
			"data/decay_log_fit.txt" using 1:2 with lines title "Fitted Decay Law", \
			"data/decay_log_fit.txt" using 1:3 with lines title "+σ_{ln(A)}", \
			"data/decay_log_fit.txt" using 1:4 with lines title "-σ_{ln(A)}", \
			"data/decay_log.txt" using 1:2:3 with yerrorbars title "Data Points", \
	' | gnuplot

plot.decay.log.sigma_lambda:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Decay Law Fit, ln(y) = ln(a) - λt"; \
		set output "plots/decay_log_fit_sigmalambda.svg"; \
		set key left ;\
		set xlabel "t";\
		set ylabel "log(y)";\
		plot  \
			"data/decay_log_fit.txt" using 1:2 with lines title "Fitted Decay Law", \
			"data/decay_log_fit.txt" using 1:5 with lines title "+σ_{lambda}", \
			"data/decay_log_fit.txt" using 1:6 with lines title "-σ_{lambda}", \
			"data/decay_log.txt" using 1:2:3 with yerrorbars title "Data Points", \
	' | gnuplot

plot.decay.sigma_A:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Decay Law Fit, y = a*exp(λt)"; \
		set output "plots/decay_fit_sigmaA.svg"; \
		set key left ;\
		set xlabel "t";\
		set ylabel "log(y)";\
		plot  \
			"data/decay_fit.txt" using 1:2 with lines title "Fitted Decay Law", \
			"data/decay_fit.txt" using 1:3 with lines title "+σ_{A}", \
			"data/decay_fit.txt" using 1:4 with lines title "-σ_{A}", \
			"data/decay.txt" using 1:2:3 with yerrorbars title "Data Points"; \
	' | gnuplot

plot.decay.sigma_lambda:
	echo ' \
		set terminal svg size 600,600 background rgb "white"; \
		set title "Decay Law Fit, y = a*exp(λt)"; \
		set output "plots/decay_fit_sigmalambda.svg"; \
		set key left ;\
		set xlabel "t";\
		set ylabel "log(y)";\
		plot  \
			"data/decay_fit.txt" using 1:2 with lines title "Fitted Decay Law", \
			"data/decay_fit.txt" using 1:5 with lines title "+σ_{lambda}", \
			"data/decay_fit.txt" using 1:6 with lines title "-σ_{lambda}", \
			"data/decay.txt" using 1:2:3 with yerrorbars title "Data Points"; \
	' | gnuplot

clean:
	rm -f *.o $(INCLUDE_DIR)/*.o main out.txt plots/*.svg 

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra
INCLUDEPATH = ../includes/

data.txt: main
	 ./$< > $@

main: main.o ../includes/sfuns.o
	c++ -o main main.o ../includes/sfuns.o

main.o: main.cpp 
	c++ -c main.cpp 


plot.erf : data.txt erf.data makefile
	echo ' \
		set terminal svg; \
		set output "erf.svg"; \
		set key left ;\
		set xlabel "x";\
		set ylabel "erf(x)";\
		plot  \
			"$<" index 0 with lines title "erf(x)" \
			,"erf.data" with points pointtype 4 title "tabulated erf(x)" \
	' | tee log.erf.gpi | gnuplot

plot.gamma : data.txt makefile
	echo ' \
		set terminal svg; \
		set output "gamma.svg"; \
		set key bottom ;\
		set xlabel "x" ;\
		set ylabel "Γ(x)" ;\
		plot [-4:][-20:25] \
			"$<" index 1 with lines title "gamma(x)" \
			,"$<" index 3 with points title "factorials" \
	' | tee log.gamma.gpi | gnuplot

plot.lngamma : data.txt makefile
	echo ' \
		set terminal svg; \
		set output "lngamma.svg"; \
		set key bottom ;\
		set xlabel "x" ;\
		set ylabel "ln(Γ(x))" ;\
		plot [-4:][-20:20] \
			"$<" index 2 with lines title "ln(Γ(x))" \
			,"$<" index 4 with points title "ln(factorials)" \
	' | tee log.lngamma.gpi | gnuplot

plot.all: plot.lngamma plot.gamma plot.erf #Runs all plot functions




clean:
	rm -f *.o out.* main *.svg *.png
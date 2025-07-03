CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra
N=1e8

out.txt times.txt: main Makefile
	> out.txt
	> times.txt
	for t in $$(seq 1 12); do \
		echo -n "$$t	" >> times.txt; \
		(/usr/bin/time --format="%e" ./main -terms $(N) -threads $$t >> out.txt) 2>> times.txt; \
	done

main: main.o 
	$(CXX) -o main main.o 

main.o: main.cpp 
	$(CXX) $(CXXFLAGS) -c main.cpp



plot.times : times.txt Makefile
	echo ' \
		set terminal svg; \
		set output "times.svg"; \
		set key left ;\
		set xlabel "Number of threads";\
		set ylabel "time [s]";\
		plot  \
			"$<" with lines title ""\
	' | tee log.times.gpi | gnuplot
	
	
clean:
	rm -f *.o [Oo]ut.txt times.txt log.times.gpi times.svg main
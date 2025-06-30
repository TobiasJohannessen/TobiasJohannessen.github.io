Out.txt: main
	./main > Out.txt

main: main.o
	g++ main.o -o main

main.o: main.cpp ../includes/sfuns.cpp
	g++ -c main.cpp



clean:
	rm -f main *.o Out.txt

run: main
	./main > Out.txt && cat Out.txt
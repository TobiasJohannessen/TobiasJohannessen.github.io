Out.txt: main
	./main > Out.txt

main: main.o
	g++ main.o -o main

main.o: main.cpp
	g++ -c main.cpp


clean:
	$(RM) main.o Out.txt
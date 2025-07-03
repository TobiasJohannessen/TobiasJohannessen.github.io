CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra
INCLUDEPATH = ../includes/

out.txt: main
	./main > out.txt

main: main.o ../includes/vec.o
	c++ -o main main.o ../includes/vec.cpp ../includes/sfuns.cpp

main.o: main.cpp ../includes/vec.o
	c++ -c main.cpp


clean:
	rm -f main.o out.txt main
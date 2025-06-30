CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

out.txt: main
	./main > out.txt

main: main.o ../includes/sfuns.cpp
	c++ -o main main.o ../includes/sfuns.cpp

main.o: main.cpp 
	c++ -c main.cpp


clean:
	rm -f main.o out.txt vec.o main
CXX = g++
#CXXFLAGS = -std=c++11 -Wall -Wextra

out.txt: main
	./main 

main: main.o matrix.o
	$(CXX) $(CXXFLAGS) -o main main.o matrix.o

main.o: main.cpp ../matrix.h
	$(CXX) $(CXXFLAGS) -c main.cpp

matrix.o: ../matrix.cpp ../matrix.h
	$(CXX) $(CXXFLAGS) -c ../matrix.cpp

clean:
	rm -f *.o out.txt main
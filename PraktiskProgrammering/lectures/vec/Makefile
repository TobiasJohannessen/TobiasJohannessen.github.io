Out.txt: main
	./main > Out.txt

main: main.o
	g++ main.o -o main

main.o: main.cpp
	g++ -c main.cpp

sfuns.o : sfuns.cpp sfuns.h
	$(CXX) $(CXXFLAGS) -c -o sfuns.o sfuns.cpp #-c stands for compile only


clean:
	rm -f main *.o Out.txt

run: main
	./main > Out.txt && cat Out.txt
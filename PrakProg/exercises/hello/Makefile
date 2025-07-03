Out.txt: main
	./main > Out.txt

main: hello.o
	$(CXX) hello.o -o main

hello.o: hello.cpp
	$(CXX) -c hello.cpp

clean:                           # a phoney target, no dependencies
	rm -f Out.txt hello.o  # remove secondary files

 
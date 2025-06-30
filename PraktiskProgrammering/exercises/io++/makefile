output.txt: main main.cpp input.txt makefile
	./main --input input.txt --output output.txt

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main

clean:
	rm -f main output.txt
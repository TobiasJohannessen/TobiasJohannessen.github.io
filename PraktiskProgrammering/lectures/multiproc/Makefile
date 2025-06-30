CXXFLAGS = -Wall -std=c++23
CODE = $(filter %.cs,$^)
LIBS = $(addprefix -reference:,$(filter %.dll,$^))
MKEXE = mcs -target:exe -out:$@ $(LIBS) $(CODE)
MKLIB = mcs -target:library -out:$@ $(LIBS) $(CODE)
TIME = time --output=$@ --append --format "$$nthreads %e %U"



output.txt: main main.cpp input.txt makefile
	./main --input input.txt --output output.txt

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main
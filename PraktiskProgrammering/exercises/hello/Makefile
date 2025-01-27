hello.out : hello.cpp
	echo c++ hello.cpp

Out.txt : hello.out
	hello.out > Out.txt

.PHONEY:clean
clean:
	rm -f Out.txt hello.out

#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<iomanip>
using namespace std;


int main(int argc, char* argv[]) {
	string infile = " "; 
	string outfile = " ";
    
	for (int i = 0; i < argc; i++) {
		string arg = argv[i];
		//cout << "arg[" << i << "] = " << arg << endl;
		if(arg=="--input" && i+1<argc){
			infile = argv[i+1];
		};
		if(arg=="--output" && i+1<argc){
			outfile = argv[i+1];
		}
	}
	//cerr << "infile = " << infile << ", outfile = " << outfile << endl;
	if (infile == " " || outfile == " ") return 0;

	ifstream instream(infile);
	ofstream outstream(outfile);

	double x;
    if (instream.fail()) {
        cerr << "Error: cannot open file " << infile << endl;
        return 1;
    }
    if (outstream.fail()) {
        cerr << "Error: cannot open file " << outfile << endl;
        return 1;
    };
	string str;
	getline(instream, str);
    outstream << fixed << "x (pi)" << "\t" << "sin(x)" << "\t" << "cos(x)" << endl;
	while(instream >> x){
		x = x/6;
        outstream << fixed << x << "\t"<< sin(x * M_PI) << "\t" << cos(x * M_PI) << endl;
    };  
	

	instream.close();
	outstream.close();
	return 0;
}

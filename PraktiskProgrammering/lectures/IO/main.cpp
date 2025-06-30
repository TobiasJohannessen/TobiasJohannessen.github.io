#include<iostream>
#include<fstream>
#include<string>
using namespace std;


int main(int argc, char* argv[]) {
	string infile = " "; 
	string outfile = " ";
	for (int i = 0; i < argc; i++) {
		string arg = argv[i];
		cout << "arg[" << i << "] = " << arg << endl;
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

	string word;
	while(instream >> word) outstream << word << endl;
	


	instream.close();
	outstream.close();
	return 0;
}
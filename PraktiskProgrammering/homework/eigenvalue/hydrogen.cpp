#include <iostream>
#include"../includes/matrix.h" // Assuming matrix.h is in the includes directory
#include <cmath>
#include<string>
#include <fstream> // Required for file operations
#include <iomanip> // Required for std::fixed and std::setprecision

using namespace pp;
int main(int argc, char* argv[]){
    double rmax = 5.0; // Default value for rmax
    double dr = 0.1; // Default value for dr
    std::string conv_type = "";
    for (int i = 0; i < argc; i++) {
		std::string arg = argv[i];
		if(arg=="-rmax" && i+1<argc){
			rmax = std::stod(argv[i+1]);
		};
		if(arg=="-dr" && i+1<argc){
			dr = std::stod(argv[i+1]);
		};
        if(arg=="-conv" && i+1<argc){
            conv_type = argv[i+1];
        };
	};
    int npoints = static_cast<int>(rmax / dr) - 1;

    //std::cout << "rmax = " << rmax << ", dr = " << dr << ", npoints = " << npoints << std::endl;

    vector r(npoints);
    for (int i = 0; i < npoints; i++) {
        r[i] = (i + 1) * dr;
    }
    //r.print("r = \n"); Print for checking

    // Construct the Hamiltonian matrix H
    
    

    matrix H = matrix(npoints,npoints);
        for(int i=0;i<npoints-1;i++){
        H(i,i)  =-2*(-0.5/dr/dr);
        H(i,i+1)= 1*(-0.5/dr/dr);
        H(i+1,i)= 1*(-0.5/dr/dr);
        }
        H(npoints-1,npoints-1)=-2*(-0.5/dr/dr);
        for(int i=0;i<npoints;i++)H(i,i)+=-1/r[i];

        //H.print("H = \n"); Print for checking

    matrix V;
    vector w;
    try {
        EVD evd = EVD(H); // Perform eigenvalue decomposition
        V = evd.V;
        w = evd.w;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during eigenvalue decomposition: " << e.what() << std::endl;
        return 1;
    }
    // Print the eigenvalues and eigenvectors
    //std::cout << "Eigenvalues (w):" << std::endl;
    //w.print("w = \n");
    //std::cout << "Eigenvectors (V):" << std::endl;
    //V.print("V = \n");

    //Calculate the 

    if (conv_type == "dr"){
        std::ofstream outputFile1("dr_e0.txt", std::ios_base::app); // Open in append mode
        

        if (outputFile1.is_open()) {
            // Use std::fixed and std::setprecision for consistent formatting
            outputFile1 << std::fixed << std::setprecision(10) << dr << "\t" << w[0] << "\n";
            outputFile1.close(); // Close the file
            //std::cout << "Ground state energy and dr written to ground_state_data.txt" << std::endl;
        } else {
            std::cerr << "Error: Unable to open file dr_e0.txt" << std::endl;
            return 1;
        }
    };

    if (conv_type == "rmax"){
        std::ofstream outputFile2("rmax_e0.txt", std::ios_base::app); //
        if (outputFile2.is_open()) {
            // Use std::fixed and std::setprecision for consistent formatting
            outputFile2 << std::fixed << std::setprecision(10) << rmax << "\t" << w[0] << "\n";
            outputFile2.close(); // Close the file
            //std::cout << "Ground state energy and rmax written to ground_state_data.txt" << std::endl;
        } else {
            std::cerr << "Error: Unable to open file rmax_e0.txt" << std::endl;
            return 1;
        }
    };


    std::ofstream outputFile3("wavefunction.txt", std::ios_base::app); // Open in append mode
    if (outputFile3.is_open()) {
        // Write the wavefunctions to the file
        for (int j = 0; j < V.size2(); j++) {
        outputFile3 << std::fixed << std::setprecision(10) << r[j] << "\t";
        for (int i = 0; i < 5; i++) {
                outputFile3 << V(j, i) * 1/(sqrt(dr)) << "\t"; // Write each component of the wavefunction
            }
            outputFile3 << "\n"; // New line for the next point
        }
        outputFile3.close(); // Close the file
        //std::cout << "Wavefunctions written to hydrogen_wavefunctions.txt" << std::endl;
    } else {
        std::cerr << "Error: Unable to open file hydrogen_wavefunctions.txt" << std::endl;
        return 1;
    }
    return 0;
};
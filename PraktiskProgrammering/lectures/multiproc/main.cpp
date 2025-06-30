#include<iostream>
#include<thread>
#include<string>

struct datum {
    int start, stop;
    double sum;
};

void harm(datum& d){
    d.sum=0;
    for(long int i = d.start+1; i<=d.stop; i++){
        d.sum+=1.0/i;
    }

};

int main(std::string argc, char* argv[]){
    int nthreads=1, nterms=(int)1e8;
    for(int i=0; i<argc;i++){
        std::string arg = argv[i];
        if(arg=="-threads" && i+1<argc) nthreads=std::stoi(argv[i+1]);
        if(arg=="-terms" && i+1<argc) nterms=(int)std:stod(argv[i+1]);
    }
    std::cerr << "nthreads=" << nthreads << " nterms=" << nterms << "\n";
    std::vector < std::thread > threads(nthread);
    std::vector < datum > std::data(nthread);
    for (int i=0; i<nthreads; i++){
        data[i].start = i*nterms/nthreads;
        data[i].stop = (i+1)*nterms/nthreads;
    };
    for (int i=0; i<nthreads; i++){
        threads[i]=std::thread(harm, std::ref(data[i]));
    };
    for(std::thread thread : threads){ thread.join();}
    double sum=0;
    for(datum d: data){sum+=d.sum;};
    std::cout << "sum=" << sum << std::endl;

    return 0;
};
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
using namespace std;

// random generator function:
int myrandom(int i) {
    return std::rand() % i;
}
int main(int argc, char **argv) {
    // get n from cli
    int n = atoi(argv[1]);
    // seed rand with time
    srand(time(0));
    // make ss for name of file
    stringstream ss;
    // put in name of file
    ss << n << "x" << n << ".txt";
    // may file stream
    ofstream myfile;
    // open file stream with name
    myfile.open(ss.str().c_str());
    // for men and women
    for (int k = 0; k < 2; k++) {
        // for n lines of file
        for (int j = 0; j < n; j++) {
            // make array of for random numbers
            int *a = new int[n];
            for (int i = 0; i < n; i++) {
                // initialize random numbers from 1-n
                a[i] = i + 1;
            }
            // shuffle numbers
            random_shuffle(a, a + n, myrandom);
            // for n items in a line print to file
            for (int i = 0; i < n; i++) {
                //cout << a[i] << ' ';
                // number followed by space
                myfile << a[i] << ' ';
            }
            //cout << endl;
            // print new line
            if (j != n - 1) {
                myfile << endl;
            }
            // free memory
            delete[] a;
        }
    if(k==0){
        myfile<<endl;
    }

    }
    // close file
    myfile.close();
}


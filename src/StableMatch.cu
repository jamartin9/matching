// for cout
#include <iostream>
// for signal handlers
#include <signal.h>
// for string stream after reading from file
#include <sstream>
// for file stream
#include <fstream>
// c++ STL for string
#include <string>
// for sleep call
#include <unistd.h>

using namespace std;

/****************** Signal Handler to stop algorithm ******************/
void MyHandler(int Signal) {
	// switch what we do for each Signal that used handler
	// TODO: print out current matching
	switch (Signal) {
	// handle SIGTERM
	case SIGTERM: {
		cerr << "Error with signal SIGTERM Caught\n" << endl;
		break;
	}
		// handle SIGINT
	case SIGINT: {
		cerr << "Error with signal SIGINT caught\n" << endl;
		break;
	}
		// default to be safe
	default: {
		cerr << "Error with signal " << Signal << " was caught\n" << endl;
		break;
	}
	}

}
/****************** End of Signal Handler ******************/

struct Element {
	// graph x value
	int x;
	// graph y value
	int y;
	// l value of the graph (man pref)
	int LValue;
	// r value of the graph (women pref)
	int RValue;
	// type of point on graph (0 = unmatched(unstable), 1= Matched(stable), 2= nm1 generating)
	int Type;
	// logical pointer to element
	int pointer;
};


int main(int argc, char **argv) {
	/****************** Setup Signal Handlers ******************/
	// declare signal actions
	struct sigaction SigInt, SigTerm;

	// clear the structure for SIGINT
	memset(&SigInt, 0, sizeof(SigInt));

	// setup SIGINT as MyHandler and check errors
	SigInt.sa_handler = MyHandler;
	sigemptyset(&SigInt.sa_mask);
	if (-1 == sigaction(SIGINT, &SigInt, NULL)) {
		cerr << "Error assigning SIGINT\n" << endl;

	}
	// clear the structure
	memset(&SigTerm, 0, sizeof(SigTerm));

	// setup SIGTERM as MyHandler and check errors
	SigTerm.sa_handler = MyHandler;
	sigemptyset(&SigTerm.sa_mask);
	if (-1 == sigaction(SIGTERM, &SigTerm, NULL)) {
		cerr << "Error assigning SIGTERM\n" << endl;
	}
	/****************** End of Signal Handlers ******************/

	/****************** Check CLI arguments ******************/

	// check that the number of arguments is two
	if (argc != 2) {
		// print error and return -1
		cerr
				<< "Incorrect Number of Arguments\nPlease Enter the Path to the file as the first argument"
				<< endl;
		return -1;
	}
	/****************** End of Checking Arguments ******************/

	/****************** Read In File For Data ******************/
	// make a pointer to the path of the file from the CLI arguments
	const char *UsersFile;
	UsersFile = argv[1];
	// make a counter for the number of line/people n
	int n = 0;
	// make a file stream called inFile from the users file
	ifstream inFile(UsersFile);
	// make string to hold file information
	string s = "";
	// if the file exists and is open
	if (inFile) {
		// read all data into string from file
		while (inFile.good()) {
			// make a string to hold each line
			string templine;
			// get the line from the file
			getline(inFile, templine);
			// add each line to the String s
			s = s + templine;
			// increment the number of participants for each line
			n++;
		}
		// close the file stream
		inFile.close();
	} else {
		cerr << "File Doesn't Exist or Couldn't be opened" << endl;
		return -1;
	}
	/****************** End Of File Reading ******************/
	// make a string stream from string read from file
	stringstream ss(s);
	// divide n by 2 because we have menPrefs and womenPrefs in the same file
	n = n / 2;
	// create element pointer for Ranking Matrix
	Element *rankingMatrix;
	// allocate memory on gpu for rankingMatrix
	cudaMallocManaged(&rankingMatrix,(sizeof(&rankingMatrix)*(n*n)));

	/************************ Move This Section to CUDA *********************************/
	// add mens prefs to the matrix by row
	// add x and y values at the same time
	// matrix is read left -> right, top -> down
	// this is generation of first arbitrary matching
	for (int i = 0, col = 0, row = 1; i < n * n; i++) {
		// initialize r Value, type and pointer
		// set r value to default to 0
		rankingMatrix[i].RValue = 0;
		// set type to default to 0
		rankingMatrix[i].Type = 0;
		// set the pointer to point to its element in the array
		rankingMatrix[i].pointer = i;
		// set the l value based on Mans preference
		ss >> rankingMatrix[i].LValue;
		// if i%n is 0 we are at the next Y position
		if (0 == i % n) {
			// increment col for a new col
			col++;
		}
		// for each col set y to that col
		rankingMatrix[i].y = col;
		// set the x value
		rankingMatrix[i].x = row;
		// if we are at the end of the matrix row
		if (row == n) {
			// reset the row
			row = 1;
		} else {
			// increment row for the next x value
			row++;
		}
	}
	// add womens prefs to the matrix by column
	for (int i = 0, colsDone = 0; i < n && colsDone < n;) {
		ss >> rankingMatrix[(i * n) + colsDone].RValue;
		i++;
		if (i == n) {
			i = 0;
			colsDone++;
		}

	}
	// print out ranking matrix for checking
	for (int i = 0; i < n * n; i++) {
		// make sure i isn't out of bounds and the Y values are the same
		if ((i + 1 < n * n) && rankingMatrix[i].y == rankingMatrix[i + 1].y) {
			cout << "(" << rankingMatrix[i].LValue << ","
					<< rankingMatrix[i].RValue << ") ";

		}
		// otherwise print out the last matching for row and a new line
		else {
			cout << "(" << rankingMatrix[i].LValue << ","
					<< rankingMatrix[i].RValue << ") " << endl;
		}
	}
	/****************** End of Matrix Generation *********/


	cudaFree(rankingMatrix);
	/****************** Reset And End ******************/
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
	// return 0 for success
	return 0;
	/****************** End Of Program ******************/
}

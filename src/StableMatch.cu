//Includes
//for cout
#include <iostream>
//For signal handlers
#include <signal.h>
//For string stream after reading from file
#include <sstream>
//For file stream
#include <fstream>
//C++ STL for string
#include <string>
//For Sleep call
#include <unistd.h>
using namespace std;
//See if this change is pushed

/****************** Signal Handler to stop algorithm ******************/
void MyHandler(int Signal) {
	//Switch what we do for each Signal that used handler
	//TODO: print out current matching
	switch (Signal) {
	//Handle SIGTERM
	case SIGTERM: {
		cerr << "Error with signal SIGTERM Caught\n" << endl;
		break;
	}
		//Handle SIGINT
	case SIGINT: {
		cerr << "Error with signal SIGINT caught\n" << endl;
		break;
	}
		//Default to be safe
	default: {
		cerr << "Error with signal " << Signal << " was caught\n" << endl;
		break;
	}
	}

}
/****************** End of Signal Handler ******************/

struct Element {
	//Graph X Value
	int x;
	//Graph Y Value
	int y;
	//L Value of the graph (Man Pref)
	int LValue;
	//R Value of the graph (Women Pref)
	int RValue;
	//Type of point on graph (0 = unmatched(unstable), 1= Matched(stable), 2= nm1 generating)
	int Type;
	//Logical pointer to element
	int pointer;
};


int main(int argc, char **argv) {
	/****************** Setup Signal Handlers ******************/
	//Declare Signal Actions
	struct sigaction SigInt, SigTerm;

	//clear the structure for SIGINT
	memset(&SigInt, 0, sizeof(SigInt));

	//Setup SIGINT as MyHandler and check errors
	SigInt.sa_handler = MyHandler;
	sigemptyset(&SigInt.sa_mask);
	if (-1 == sigaction(SIGINT, &SigInt, NULL)) {
		cerr << "Error assigning SIGINT\n" << endl;

	}
	//clear the structure
	memset(&SigTerm, 0, sizeof(SigTerm));

	//Setup SIGTERM as MyHandler and check errors
	SigTerm.sa_handler = MyHandler;
	sigemptyset(&SigTerm.sa_mask);
	if (-1 == sigaction(SIGTERM, &SigTerm, NULL)) {
		cerr << "Error assigning SIGTERM\n" << endl;
	}
	/****************** End of Signal Handlers ******************/

	/****************** Check CLI arguments ******************/

	//Check that the number of arguments is two
	if (argc != 2) {
		//Print Error and return -1 for error
		cerr
				<< "Incorrect Number of Arguments\nPlease Enter the Path to the file as the first argument"
				<< endl;
		return -1;
	}
	/****************** End of Checking Arguments ******************/

	/****************** Read In File For Data ******************/
	//Make a pointer to the path of the file from the CLI arguments
	const char *UsersFile;
	UsersFile = argv[1];
	//Make a counter for the number of line/people n
	int n = 0;
	//Make a file stream called inFile from the users file
	ifstream inFile(UsersFile);
	//Make string to hold file information
	string s = "";
	//If the file exists and is open
	if (inFile) {
		//Read all data into string from file
		while (inFile.good()) {
			//Make a string to hold each line
			string templine;
			//get the line from the file
			getline(inFile, templine);
			//Add each line to the String s
			s = s + templine;
			//increment the number of participants for each line
			n++;
		}
		//Close the file stream
		inFile.close();
	} else {
		cerr << "File Doesn't Exist or Couldn't be opened" << endl;
		return -1;
	}
	/****************** End Of File Reading ******************/
	//Make a string stream from string read from file
	stringstream ss(s);
	//Divide n by 2 because we have menPrefs and womenPrefs in the same file
	n = n / 2;
	//Make Ranking Matrix of size n^2
	//Element rankingMatrix[n * n];

	//Create element pointer for Ranking Matrix
	Element *rankingMatrix;
	//Allocate memory on gpu for rankingMatrix
	//cudaMallocManaged(&rankingMatrix,(n*n));
	cudaMallocManaged(&rankingMatrix,(sizeof(&rankingMatrix)*(n*n)));

	/************************ Move This Section to CUDA *********************************/
	//Add Mens Prefs to the matrix by Row
	//Add X and Y values at the same time
	//Matrix is read left -> right, top -> down
	//This is generation of first arbitrary matching
	for (int i = 0, col = 0, row = 1; i < n * n; i++) {
		//initialize R Value, type and pointer
		//set R value to default to 0
		rankingMatrix[i].RValue = 0;
		//Set type to default to 0
		rankingMatrix[i].Type = 0;
		//Set the pointer to point to its element in the array
		rankingMatrix[i].pointer = i;
		//Set the L value based on Mans preference
		ss >> rankingMatrix[i].LValue;
		//If i%n is 0 we are at the next Y position
		if (0 == i % n) {
			//increment col for a new col
			col++;
		}
		//For each col set y to that col
		rankingMatrix[i].y = col;
		//set the x value
		rankingMatrix[i].x = row;
		//If we are at the end of the matrix row
		if (row == n) {
			//reset the row
			row = 1;
		} else {
			//Increment row for the next x value
			row++;
		}
	}
	//Add Womens Prefs to the matrix by column
	for (int i = 0, colsDone = 0; i < n && colsDone < n;) {
		ss >> rankingMatrix[(i * n) + colsDone].RValue;
		i++;
		if (i == n) {
			i = 0;
			colsDone++;
		}

	}
	//Print out Ranking Matrix for checking
	for (int i = 0; i < n * n; i++) {
		//Make sure i isn't out of bounds and the Y values are the same
		if ((i + 1 < n * n) && rankingMatrix[i].y == rankingMatrix[i + 1].y) {
			cout << "(" << rankingMatrix[i].LValue << ","
					<< rankingMatrix[i].RValue << ") ";

		}
		//Otherwise print out the last matching for row and a new line
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
	//Return 0 for success
	return 0;
	/****************** End Of Program ******************/
}

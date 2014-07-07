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
// for printf()
#include <cstdio>
// for time()
#include <time.h>
// for sleep()
//#include <unistd.h>

// use standard namespace
using namespace std;

// structure for each Processor Element
struct Element {
	// graph x value, column
	int x;
	// graph y value, row
	int y;
	// l value of the graph (man pref)
	int LValue;
	// r value of the graph (women pref)
	int RValue;
	// type of point on graph
	// (0 = unmatched(unstable), 1= Matched(stable), 2= nm1 generating)
	int Type;
	// logical pointer to element
	int pointer;
};

// structure to represent a man woman pair
struct Pair {
	// integer to hold element a, e.g. the man
	int a;
	// integer to hold element b, e.g. the woman
	int b;
};

// global pointer on host to represent pair
Pair *pairs;

// make global a counter for the number of lines/people
int n = 0;

// function for printing pairs
void printPairs() {

	// check that n isn't 0
	if (0 == n) {
		// print error
		cerr << "No participants" << endl;
	} else {
		// for each pair
		for (int i = 0; i < n; i++) {
			// print out both values
			cout << "( " << pairs[i].a << " , " << pairs[i].b << " ) ";
		}
		cout << endl;
	}
}

// signal handler function to stop algorithm
void signalHandler(int signal) {

	// switch what we do for each signal that used handler
	switch (signal) {
	// handle SIGTERM
	case SIGTERM: {
		printPairs();
		break;
	}
		// handle SIGINT
	case SIGINT: {
		printPairs();
		break;
	}
		// default to be safe
	default: {
		printPairs();
		break;
	}
	}

}

// function to create signal handlers
void createSignalHandlers() {

	// declare signal actions
	struct sigaction SigInt, SigTerm;

	// clear the structure for SIGINT
	memset(&SigInt, 0, sizeof(SigInt));

	// setup SIGINT as signalHandler and check errors
	SigInt.sa_handler = signalHandler;
	sigemptyset(&SigInt.sa_mask);
	if (-1 == sigaction(SIGINT, &SigInt, NULL)) {
		cerr << "Error assigning SIGINT\n" << endl;

	}
	// clear the structure
	memset(&SigTerm, 0, sizeof(SigTerm));

	// setup SIGTERM as signalHandler and check errors
	SigTerm.sa_handler = signalHandler;
	sigemptyset(&SigTerm.sa_mask);
	if (-1 == sigaction(SIGTERM, &SigTerm, NULL)) {
		cerr << "Error assigning SIGTERM\n" << endl;
	}
}

// CUDA kernel to create n disjoint lists over the columns of the ranking
// matrix such that
//     PE_(i,j) points to PE_(i_1,j)
// (Lu2003, 47).
__global__ void preMatch(Element *rankingMatrix, int n) {

	// if the thread is not out of bounds
	if (threadIdx.x + n < n * n) {
		// set the current PE's pointer to the next PE
		rankingMatrix[threadIdx.x].pointer =
				rankingMatrix[threadIdx.x + n].pointer;
	}
}

// CUDA kernel to swap a PE's pointer to another random PE's pointer that is in
// the same row such that
//     PE_(i,i) points to PE_(i+1,j) and
//     PE_(i,j) points to PE_(i+1,i),
//     where i <= j <= n,
// forming n new disjoit lists starting at PE_(1,j) (Lu2003, 47).
__global__ void perMatch(Element *rankingMatrix, int n, int randomOffsets[]) {

	//printf("randomOffsets[threadIdx.x]: %i\n", randomOffsets[threadIdx.x]);

	// save index of current element
	int pos1 = n * threadIdx.x + threadIdx.x;

	// save index of element in same row we are swapping with
	int pos2 = pos1 + randomOffsets[threadIdx.x];

	//printf("pos1: %i, pos2: %i\n", pos1, pos2);

	// save current pointer
	int pointer = rankingMatrix[pos1].pointer;

	// switch pointers with another element in same row
	rankingMatrix[pos1].pointer = rankingMatrix[pos2].pointer;

	// set other elements point to old pointer
	rankingMatrix[pos2].pointer = pointer;

}

// CUDA kernel to create an initial matching and store it in pairs by following
// the singly linked list generated by perMatch(). "Each PE_(1,j) finds the
// other end PE_(n,p(1,j)) of its list where p(1,j) is the column position of
// the PE pointed to by PE_(1,j). Hence, a matching {(j,p(1,j))|1<=j<=n} is
// formed" (Lu2003, 47).
__global__ void iniMatch(Element *rankingMatrix, int n, Pair pairs[]) {

	// make temporary element to hold first item
	Element element = rankingMatrix[threadIdx.x];

	// take x value for one member of pair
	pairs[threadIdx.x].a = element.x;

	// go to the end of the list
	for (int i = 0; i < n - 1; i++) {

		//change element to what each element points to
		element = rankingMatrix[element.pointer];
	}

	// take the x value of the last element pointed to after following list
	// for the second member of the pair
	pairs[threadIdx.x].b = element.x;

	// print out thread and pairs value
	//printf("(%i,%i)\n", pairs[threadIdx.x].a, pairs[threadIdx.x].b);

}

// function to print out ranking matrix L and R values
void printRankingMatrix(Element rankingMatrix[]) {

	// for each element of the ranking matrix
	for (int i = 0; i < n * n; i++) {

		// make sure i isn't out of bounds and the Y values are the same (same row)
		if ((i + 1 < n * n) && rankingMatrix[i].y == rankingMatrix[i + 1].y) {

			// print out L and R value
			cout << "(" << rankingMatrix[i].LValue << ","
					<< rankingMatrix[i].RValue << ") ";

		}

		// otherwise print out the last PE for the row and a new line for the next row
		else {
			// print out last pair in row and start new row
			cout << "(" << rankingMatrix[i].LValue << ","
					<< rankingMatrix[i].RValue << ") " << endl;
		}
	}

	// print out a new line
	cout << endl;
}

// function to print out ranking matrix pointers
void printRankingMatrixPointers(Element rankingMatrix[]) {

	// print out Ranking Matrix for checking
	for (int i = 0; i < n * n; i++) {

		// make sure i isn't out of bounds and the Y values are the same
		if ((i + 1 < n * n) && rankingMatrix[i].y == rankingMatrix[i + 1].y) {
			cout << rankingMatrix[i].pointer << " ";
		}

		// otherwise print out the last matching for row and a new line
		else {
			cout << rankingMatrix[i].pointer << endl;
		}
	}
	cout << endl;
}

// function to generate random numbers for offsets of pointer swapping
void generateRandomOffsets(int randomOffsets[]) {

	//TODO: Use cuRand for random number generation

	// get random seed
	srand(time(NULL));

	// for the number of random offsets we need (n-1)
	for (int i = n - 1, j = 0; i > 0; i--, j++) {

		// initialize variable for the random number
		int randInt = -1;
		do {
			// generate a random number
			randInt = (rand() % n) + 1; // 1 <= randInt <= n
		}

		// keep generating random numbers while the number isn't within range
		while (randInt > i);

		// set the random off set to the random number
		randomOffsets[j] = randInt;

		// print out the number generated
		//cout << randomOffsets[j] << " ";
	}
}

// function to generate initial match on host
void initMatch(Element rankingMatrix[]) {

	// call kernel with n^2 threads for creating disjoint lists with pointers
	preMatch<<<1, n * n>>>(rankingMatrix, n);

	// synchronize with the device
	cudaDeviceSynchronize();

	//printRankingMatrixPointers(rankingMatrix, n);

	// create array for random offsets
	int * randomOffsets;

	// allocate random offsets array on GPU
	cudaMallocManaged(&randomOffsets, (sizeof(randomOffsets[0]) * (n - 1)));

	// call function to create random numbers
	generateRandomOffsets(randomOffsets);

	// call kernel to do pointer swapping
	perMatch<<<1, n - 1>>>(rankingMatrix, n, randomOffsets);

	// synchronize with the device
	cudaDeviceSynchronize();

	//printRankingMatrixPointers(rankingMatrix, n);

	// free the random off sets on the GPU
	cudaFree(randomOffsets);

	// allocate pairs on the GPU
	cudaMallocManaged(&pairs, sizeof(pairs[0]) * n);

	// call kernel to create initial match
	iniMatch<<<1, n>>>(rankingMatrix, n, pairs);

	// synchronize with the device
	cudaDeviceSynchronize();

	// free the pairs
	//cudaFree(pairs);
}

int main(int argc, char **argv) {

	// setup signal handlers for stopping algorithm
	createSignalHandlers();

	// check CLI arguments
	// check that the number of arguments is two
	if (argc != 2) {
		// print error and return -1
		cerr
			<< "Incorrect number of arguments\n"
			<< "Please enter the path to the file as the first argument"
			<< endl;
		return -1;
	}

	/****************** Read In File For Data ******************/

	// make a pointer to the path of the file from the CLI arguments
	const char *UsersFile;
	UsersFile = argv[1];

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
	}

	// print error and return -1 for error
	else {
		cerr << "File doesn't exist or couldn't be opened" << endl;
		return -1;
	}

	/****************** End Of File Reading ******************/

	// make a string stream from string read from file
	stringstream ss(s);

	// divide n by 2 because we have menPrefs and womenPrefs in the same file
	n = n / 2;

	// create element pointer for Ranking Matrix
	Element *rankingMatrix;

	// allocate memory on GPU for rankingMatrix
	cudaMallocManaged(&rankingMatrix, (sizeof(rankingMatrix) * (n * n)));

	//TODO: Move creation of Matrix to CUDA
	// create ranking matrix
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
		// read in each preference into column
		ss >> rankingMatrix[(i * n) + colsDone].RValue;

		// increment i
		i++;

		// if we are at the end of the column
		if (i == n) {
			// reset i
			i = 0;

			//increment the number of columns done
			colsDone++;
		}

	}

	//printRankingMatrix(rankingMatrix, n);

	// create Initial Matching by calling function
	initMatch(rankingMatrix);

	//printRankingMatrixPointers(rankingMatrix, n);

	// free the ranking matrix from the GPU
	cudaFree(rankingMatrix);

	//print out pairs (matching)
	printPairs();

	// free pairs from GPU
	cudaFree(pairs);

	/****************** Reset And End ******************/
	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits -nvidia
	cudaDeviceReset();

	// return 0 for successful completion of the program
	return 0;
}

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
// for thrust pairs
#include <thrust/pair.h>
// for time
#include <sys/time.h>

using namespace std;
//TODO:
// Make random number generation parallel
// Make matrix generation parallel
// Maybe change int Type to series of boolean values
// Allocate n on GPU and pass pointer to it when needed
// Make scale to larger than 1024 x 1024 problem size

// structure for each Processor Element
struct Element {
    // graph x value, row
    int x;
    // graph y value, column
    int y;
    // l value of the graph (e.g., man pref)
    int l;
    // r value of the graph (e.g., women pref)
    int r;
    // logical pointer to element
    Element *pointer;
    // type of the pair
    //   0: unmatched
    //   1: matched
    //   2: unstable
    //   3: nm_1-generating
    //   4: nm_1
    //   5: nm_2
    int type;
    // nm_2-generating
    bool nm2gen;
    // nm_2-generating chain r-pointer
    Element *rPointer;
    // nm_2-generating chain c-pointer
    Element *cPointer;
};

// first n pairs:
// first integer to hold element a, e.g. the man, 1 <= a <= n
// second integer to hold element b, e.g. the woman, 1 <= b <= n
//
// second n pairs:
// first integer to hold element a, e.g. the man, 1 <= a <= n
// second integer to hold element b, e.g. the woman, 1 <= b <= n
// second half of array contains swapped values sorted by key
thrust::pair<int, int> *matchingPairs;

// global state for stable matching
// true if no unstable matches exist, false otherwise
bool *stable;

// global counter for the number of lines/people
int n;

void checkArgs(int argc) {
    if (argc != 2) {
        cerr << "Incorrect number of arguments" << endl
             << "Please enter the path to the file as the first argument" << endl;
        exit(EXIT_FAILURE);
    }
}

// function for printing pairs
void printPairs() {

    // check that n isn't 0
    if (0 == n) {
        // print error
        cerr << "No participants" << endl;
    } else {
        cout << "current matching: ";
        // for each pair
        for (int i = 0; i < n; i++) {
            // print out both values
            cout << "(" << matchingPairs[i].first << ","
                        << matchingPairs[i].second << ") ";
        }
        cout << endl;
    }
}

// signal handler function to stop algorithm
void signalHandler(int signal) {

    switch (signal) {
        case SIGTERM:
        case SIGINT:
        default: {
                     printPairs();
                     exit(EXIT_SUCCESS);
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
__global__ void preMatch(Element *rankingMatrix) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;
    // if the thread is not out of bounds
    if (tid + blockDim.x < blockDim.x * blockDim.x) {
        // set the current PE's pointer to the next PE
        rankingMatrix[tid].pointer = &rankingMatrix[tid + blockDim.x];
    }
}

// CUDA kernel to swap a PE's pointer to another random PE's pointer that is in
// the same row such that
//     PE_(i,i) points to PE_(i+1,j) and
//     PE_(i,j) points to PE_(i+1,i),
//     where i <= j <= n,
// forming n new disjoint lists starting at PE_(1,j) (Lu2003, 47).
__global__ void perMatch(Element *rankingMatrix, int randomOffsets[], int n) {

    // save index of current element
    int pos1 = n * threadIdx.x + threadIdx.x;

    // save index of element in same row we are swapping with
    int pos2 = pos1 + randomOffsets[threadIdx.x];

    // save current pointer
    Element *pointer = rankingMatrix[pos1].pointer;

    // switch pointers with another element in same row
    rankingMatrix[pos1].pointer = rankingMatrix[pos2].pointer;

    // set other elements point to old pointer
    rankingMatrix[pos2].pointer = pointer;

}

// CUDA kernel to create an initial matching and store it in pairs by following
// the singly linked list (using pointer jumping) generated by perMatch(). "Each PE_(1,j) finds the
// other end PE_(n,p(1,j)) of its list where p(1,j) is the column position of
// the PE pointed to by PE_(1,j). Hence, a matching {(j,p(1,j))|1<=j<=n} is
// formed" (Lu2003, 47).
__global__ void iniMatch(Element *rankingMatrix,
                         thrust::pair<int, int> pairs[]) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    Element *e = &rankingMatrix[tid];

    // value holders for first row
    int a;
    // if we are in the first row save out pointer
    if (tid < blockDim.x) {
        // make temporary element to hold first item

        a = (*e).y; // a, as in Pair (a,b)

        // take x value for one member of pair
        pairs[tid].first = a;
    }
    // while we are not at the end
    while ( (*((*e).pointer)).x < blockDim.x) {
        // set our pointer to the pointer of the element we point to
        (*e).pointer = (*((*e).pointer)).pointer;
    }
    // if we started in the first row save our end position
    if (tid < blockDim.x) {
        int b;
        b = (*((*e).pointer)).y; // b, as in Pair (a,b)

        rankingMatrix[(a - 1) * blockDim.x + (b - 1)].type = 1; // mark as matching

        // take the x value of the last element pointed to after following list
        // for the second member of the pair
        pairs[tid].second = b;
        pairs[blockDim.x + b - 1].second = a; // for reverse lookup
        pairs[blockDim.x + b - 1].first = b; // for reverse lookup
    }
}

// CUDA kernel to mark unstable pairs in the ranking matrix
__global__ void unstable(Element *rankingMatrix,
                         thrust::pair<int, int> *pairs,
                         bool *stable) {


    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    // element associated with this thread

    Element *e = &rankingMatrix[tid];

    // element of matching pair in same row

    // get the current element's row
    int rowIndex = (*e).x - 1;

    // get current row in the ranking matrix
    int indexToRowInRankingMatrix = rowIndex * blockDim.x;

    // get the pair in the elements rows column
    // pairs[i], 0 <= i < n, sorted by rowIndex
    int colIndex = pairs[rowIndex].second - 1;

    // index is the row shifted by the column
    int index = indexToRowInRankingMatrix + colIndex;

    Element r = rankingMatrix[index];

    // element of matching pair in same column

    // get the elements current column
    colIndex = (*e).y - 1;

    // get the pair in that columns row
    // pairs[i], n <= i < 2n, sorted by colIndex
    rowIndex = pairs[blockDim.x + colIndex].second - 1;

    //get the row index of the pair
    indexToRowInRankingMatrix = rowIndex * blockDim.x;

    // index is the row shifted by the column
    index = indexToRowInRankingMatrix + colIndex;

    Element c = rankingMatrix[index];

    if (r.l > (*e).l && c.r > (*e).r) {
        rankingMatrix[tid].type = 2;
        *stable = false;
        (*e).type = 2; // mark it as unstable
    }
}

// CUDA kernel to gather attributes of potential nm_1-generating pairs
__global__ void protoNM1Gen(Element *rankingMatrix,
                            thrust::pair<int, thrust::pair<int, int> > *nm1GenPairs) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    if (rankingMatrix[tid].type == 2) { // unstable
        // take the l and col position
        nm1GenPairs[tid].first = rankingMatrix[tid].l;
        nm1GenPairs[tid].second.first = rankingMatrix[tid].x;
        nm1GenPairs[tid].second.second = rankingMatrix[tid].y;
    } else {
        // set the l value to more than the max l value
        nm1GenPairs[tid].first = blockDim.x + 1;
        nm1GenPairs[tid].second.first = rankingMatrix[tid].x;
        nm1GenPairs[tid].second.second = rankingMatrix[tid].y;
    }
}

// CUDA kernel to gather attributes of potential nm_1 pairs
__global__ void protoNM1(Element *rankingMatrix,
                         thrust::pair<int, thrust::pair<int, int> > *nm1GenPairs) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    // get the column index in ranking matrix
    int col = (rankingMatrix[tid].y - 1) * blockDim.x;

    // get the position in column
    int shift = rankingMatrix[tid].x - 1;

    //if the type is nm1 gen
    if (rankingMatrix[tid].type == 3) {

        // take the r and row position
        nm1GenPairs[col + shift].first = rankingMatrix[tid].r;
        nm1GenPairs[col + shift].second.first = rankingMatrix[tid].x;
        nm1GenPairs[col + shift].second.second = rankingMatrix[tid].y;

    } else {

        // set the r value to more than the max r value
        nm1GenPairs[col + shift].first = blockDim.x + 1;
        nm1GenPairs[col + shift].second.first = rankingMatrix[tid].x;
        nm1GenPairs[col + shift].second.second = rankingMatrix[tid].y;
    }
}

// CUDA kernel to initialize nm1Pairs entries to false
__global__ void nm1False(bool *nm1Pairs) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;
    nm1Pairs[tid] = false;
}

// CUDA kernel to mark nm2-generating pairs
// For each PE(i,j) containing a nm1-pair
//     mark the PE(l,k) as a nm2-generating pair,
//     where l = M(C(j))x, the row of the matching pair in column j and
//           k = M(R(i))y, the column of the matching pair in row i
__global__ void nm2GenDevice(Element *rankingMatrix,
                             thrust::pair<int, int> *matchingPairs,
                             Element **nm2GenPairPointers) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    // save a pointer to the Element at row i and column j in the ranking matrix
    Element *e_ij = &rankingMatrix[tid];

    // if Element is an nm_1 pair
    if ((*e_ij).type == 4) {

        int col = (*e_ij).y;
        int row = (*e_ij).x;

        // get the row of the matching pair in the same column
        int l = matchingPairs[blockDim.x + col - 1].second;

        // get the column of the matching pair in the same row
        int k = matchingPairs[row - 1].second;

        // save a pointer to the Element at row l and column k in the ranking matrix
        Element *e_lk = &rankingMatrix[blockDim.x * (l - 1) + (k - 1)];

        // if e_{l,k} is not an nm_1 pair
        if ((*e_lk).type != 4) {

            // mark it as nm2-generating
            (*e_lk).nm2gen = true;

            // save a pointer to it in the nm2GenPairPointersPointers array
            nm2GenPairPointers[l - 1] = e_lk;
            nm2GenPairPointers[blockDim.x + k - 1] = e_lk; // for reverse lookup
        }
    }
}

// CUDA kernel to set up nm2-generating chain pointers
__global__ void nm2GenPointers(Element *rankingMatrix,
                               thrust::pair<int,int> *matchingPairs,
                               bool *nm1Pairs,
                               Element **nm2GenPairPointers) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    if (rankingMatrix[tid].nm2gen) {

        Element *e = &rankingMatrix[tid];

        int row = (*e).x;
        int col = (*e).y;

        // if there is an nm_1 pair in row i
        if (nm1Pairs[row - 1] == true) {

            // the r-pointer of PE_{i,j} points to
            // the PE containing an nm_2-generating pair
            // in the same column as the matching pair in the same row as PE_{i,j}
            thrust::pair<int,int> *matchingPairInSameRow = &matchingPairs[row - 1];
            int columnOfMatchingPairInSameRow = (*matchingPairInSameRow).second;
            (*e).rPointer = nm2GenPairPointers[blockDim.x + columnOfMatchingPairInSameRow - 1];
        }
        // otherwise
        else {

            // it points to itself
            (*e).rPointer = e;
        }

        // if there is an nm_1 pair in column j
        if (nm1Pairs[blockDim.x + col - 1] == true) {
            // the c-pointer of PE_{i,j} points to
            // the PE containing an nm_2-generating pair
            // in the same row as the matching pair in the same column as PE_{i,j}

            thrust::pair<int,int> *matchingPairInSameColumn = &matchingPairs[blockDim.x + col - 1];
            int rowOfMatchingPairInSameColumn = (*matchingPairInSameColumn).second;
            (*e).cPointer = nm2GenPairPointers[rowOfMatchingPairInSameColumn - 1];
        }
        // otherwise
        else {
            // it points to itself
            (*e).cPointer = e;
        }
    }
}

// CUDA kernel to mark nm2 pairs
__global__ void nm2Pairs(Element *rankingMatrix) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    // if the pair is nm2
    if (rankingMatrix[tid].nm2gen) {
        // save pointer to element
        Element *e = &rankingMatrix[tid];
        // if the pair is row end
        int row = blockDim.x + 1;
        if (((*(*e).rPointer).x == (*e).x) && ((*(*e).rPointer).y == (*e).y)) {

            // if the pair is column end
            if (((*(*e).cPointer).x == (*e).x)
                    && ((*(*e).cPointer).y == (*e).y)) {
                // mark pair as isolated nm2
                (*e).type = 5;
                // return the stream
                return;
            }
            // otherwise save row position
            else {
                row = (*e).x;
            }
        }
        // find column end with pointer jumps (This mutates cPointer...)
        for (int i = 0; i < blockDim.x / 2; i++) {
            // if what we are pointing to points to itself
            if ((*(*e).cPointer).cPointer == (*e).cPointer) {
                // get column from c pointer if the row end reaches column end
                if (row != blockDim.x + 1) {
                    int col = (*(*e).cPointer).y;
                    // change pair to nm2
                    rankingMatrix[(blockDim.x * (row - 1)) + (col - 1)].type = 5;
                    return;
                }
                // return once thread reaches column end
                return;
            }
            // update c pointer to what we are pointing to's pointer
            (*e).cPointer = (*(*e).cPointer).cPointer;
        }
    }

}

// CUDA kernel to remove old pairs and insert new ones
__global__ void newMatching(
        Element *rankingMatrix,
        thrust::pair<int, int> *matchingPairs) {

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    // if the element is a nm pair
    if (rankingMatrix[tid].type == 4
            || rankingMatrix[tid].type == 5) {
        // get the row and col of the element
        int rowForMatching = rankingMatrix[tid].x;
        int colForMatching = rankingMatrix[tid].y;
        // change the matching pairs at row and save old value
        int rowOldPair = matchingPairs[rowForMatching - 1].first;
        matchingPairs[rowForMatching - 1].first = rowForMatching;

        int colOldPair = matchingPairs[rowForMatching - 1].second;
        matchingPairs[rowForMatching - 1].second = colForMatching;

        // change the matching pairs at the colForMatching
        matchingPairs[blockDim.x + colForMatching - 1].first = colForMatching;
        matchingPairs[blockDim.x + colForMatching - 1].second = rowForMatching;

        // update the type of new matching pairs
        rankingMatrix[tid].type = 1;

        // update the type of old matching pairs
        rankingMatrix[blockDim.x * (rowOldPair - 1) + (colOldPair - 1)].type = 0;

    }

}

// CUDA kernel to reset after one run of the ITERATION PHASE
__global__ void resetAfterIteration(Element *rankingMatrix){

    int tid = blockIdx.x*blockDim.x +threadIdx.x;
    //reset pointers to point to themselves
    rankingMatrix[tid].cPointer = &rankingMatrix[tid];
    rankingMatrix[tid].rPointer = &rankingMatrix[tid];
    rankingMatrix[tid].pointer = &rankingMatrix[tid];

    // reset nm2gen
    rankingMatrix[tid].nm2gen = false;

    // reset the type of all non matching pairs
    if(rankingMatrix[tid].type != 1){
        rankingMatrix[tid].type = 0;
    }
}

// CUDA kernel to reset all elements
__global__ void resetAll(Element *rankingMatrix,
                         thrust::pair<int,int> *matchingPairs){

    int tid = blockIdx.x*blockDim.x +threadIdx.x;

    //reset pointers to point to themselves
    rankingMatrix[tid].cPointer = &rankingMatrix[tid];
    rankingMatrix[tid].rPointer = &rankingMatrix[tid];
    rankingMatrix[tid].pointer = &rankingMatrix[tid];

    // reset nm2gen
    rankingMatrix[tid].nm2gen = false;

    // reset the type of all non matching pairs
    rankingMatrix[tid].type = 0;
    // reset the matching pairs
    if(tid < 2*blockDim.x){
        matchingPairs[tid].first = 0;
        matchingPairs[tid].second = 0;
    }
}

// nm1Gen reduction kernel
__global__ void nm1GenDevice(Element *rankingMatrix,
                             thrust::pair<int, thrust::pair<int, int> > *sharedArray,
                             int n) {

    //(blockDim*2) = n
    // thread id
    int tid = threadIdx.x;
    // tc is thread count
    for(int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1){
        // if thread is allowed to write
        if(tid < tc){
            int pa = tid * stepSize * 2;

            int pb = pa + stepSize;

            // if pa is greater than pb save pb values in pa
            if(sharedArray[blockIdx.x*n +pa].first > sharedArray[blockIdx.x*n +pb].first){
                sharedArray[blockIdx.x*n +pa].first = sharedArray[blockIdx.x*n +pb].first;
                sharedArray[blockIdx.x*n +pa].second.first = sharedArray[blockIdx.x*n +pb].second.first;
                sharedArray[blockIdx.x*n +pa].second.second = sharedArray[blockIdx.x*n +pb].second.second;
            }
        }
        __syncthreads();
    }
    __syncthreads();
    // if thread 0
    if(threadIdx.x == 0){
        // if min value isn't invalid
        if(sharedArray[blockIdx.x*n].first != n+1){
            // change type in matrix
            rankingMatrix[ n * (sharedArray[blockIdx.x*n].second.first -1) + sharedArray[blockIdx.x*n].second.second - 1 ].type = 3;
        }
    }

}

// nm1 reduction kernel
__global__ void nm1Device(Element *rankingMatrix,
                          thrust::pair<int, thrust::pair<int, int> > *sharedArray,
                          bool *nm1Pairs,
                          int n) {

    // thread id
    int tid = threadIdx.x;
    // tc is thread count
    for(int tc = blockDim.x, stepSize = 1; tc > 0; tc >>= 1, stepSize <<= 1){
        // if thread is allowed to write
        if(tid < tc){
            int pa = tid * stepSize * 2;

            int pb = pa + stepSize;

            // if pa is greater than pb save pb values in pa
            if(sharedArray[blockIdx.x*n +pa].first > sharedArray[blockIdx.x*n +pb].first){
                sharedArray[blockIdx.x*n +pa].first = sharedArray[blockIdx.x*n +pb].first;
                sharedArray[blockIdx.x*n +pa].second.first = sharedArray[blockIdx.x*n +pb].second.first;
                sharedArray[blockIdx.x*n +pa].second.second = sharedArray[blockIdx.x*n +pb].second.second;
            }
        }
        __syncthreads();
    }
    __syncthreads();
    // if thread 0
    if(threadIdx.x == 0){
        // if min value isn't invalid
        if(sharedArray[blockIdx.x*n].first != n+1){
            // change type in matrix
            rankingMatrix[ n * (sharedArray[blockIdx.x*n].second.first -1) + sharedArray[blockIdx.x*n].second.second - 1 ].type = 4;
            // remember that there is an nm_1 pair in this row
            nm1Pairs[sharedArray[blockIdx.x*n].second.first - 1] = true;

            // remember that there is an nm_1 pair in this column
            nm1Pairs[n + sharedArray[blockIdx.x*n].second.second - 1] = true;
        }
    }
}

// function to print out ranking matrix L and R values
void printRankingMatrix(Element rankingMatrix[]) {

    printf("key: pair (a,b[:] type), where type is:\n");
    printf("  0: unmatched pair\n");
    printf("  1: matched pair\n");
    printf("  2: unstable pair\n");
    printf("  3: nm1-generating pair\n");
    printf("  4: nm1 pair\n");
    printf("  5: nm2 pair\n");
    printf("and where : indicates nm2-generating pair\n");

    // for each element of the ranking matrix
    for (int i = 0; i < n * n; i++) {
        // make sure i isn't out of bounds and the Y values are the same (same row)
        //if ((i + 1 < n * n) && rankingMatrix[i].x == rankingMatrix[i + 1].x) {
        if ((i + 1) % n != 0) {
            //  if PE is nm2gen
            if (rankingMatrix[i].nm2gen == true) {
                // print out L and R value and nm2gen symbol :
                printf("(%2i,%2i %i: %2i,%2i %2i,%2i) ",
                        rankingMatrix[i].l,
                        rankingMatrix[i].r,
                        rankingMatrix[i].type,
                        ( *( rankingMatrix[i].rPointer ) ).x,
                        ( *( rankingMatrix[i].rPointer ) ).y,
                        ( *( rankingMatrix[i].cPointer ) ).x,
                        ( *( rankingMatrix[i].cPointer ) ).y);
            } else {

                // print out L and R value
                printf("(%2i,%2i %i             ) ",
                        rankingMatrix[i].l,
                        rankingMatrix[i].r,
                        rankingMatrix[i].type);
            }
        }
        // otherwise print out the last PE for the row and a new line for the next row
        else {
            //  if PE is nm2gen
            if (rankingMatrix[i].nm2gen == true) {
                // print out last pair in row and start new row
                printf("(%2i,%2i %i: %2i,%2i %2i,%2i)\n",
                        rankingMatrix[i].l,
                        rankingMatrix[i].r,
                        rankingMatrix[i].type,
                        ( *( rankingMatrix[i].rPointer ) ).x,
                        ( *( rankingMatrix[i].rPointer ) ).y,
                        ( *( rankingMatrix[i].cPointer ) ).x,
                        ( *( rankingMatrix[i].cPointer ) ).y);
            } else {

                // print out last pair in row and start new row
                printf("(%2i,%2i %i             )\n",
                        rankingMatrix[i].l,
                        rankingMatrix[i].r,
                        rankingMatrix[i].type);
            }
        }
    }
}

// function to print out ranking matrix pointers
void printRankingMatrixPointers(Element rankingMatrix[]) {

    // print out Ranking Matrix for checking
    for (int i = 0; i < n * n; i++) {

        // make sure i isn't out of bounds and the Y values are the same
        if ((i + 1 < n * n) && rankingMatrix[i].x == rankingMatrix[i + 1].x) {
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

    // get random seed
    //srand(time(NULL));
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
        //cout <<"Offsets: " <<randomOffsets[j] << " ";
    }
    //cout<<endl;
}

// function to generate initial match on host
void initMatch(Element rankingMatrix[]) {

    // call kernel with n^2 threads for creating disjoint lists with pointers
    preMatch<<<n, n>>>(rankingMatrix);

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
    perMatch<<<1, n - 1>>>(rankingMatrix, randomOffsets, n);

    // synchronize with the device
    cudaDeviceSynchronize();

    //printRankingMatrixPointers(rankingMatrix, n);

    // free the random off sets on the GPU
    cudaFree(randomOffsets);

    // allocate pairs on the GPU, n for forward, n for reverse
    //cudaMallocManaged(&matchingPairs, sizeof(matchingPairs[0]) * 2 * n);

    // call kernel to create initial match
    iniMatch<<<n, n>>>(rankingMatrix, matchingPairs);
    // synchronize with the device
    cudaDeviceSynchronize();
}

// function to mark the nm_1-generating pairs in the ranking matrix
void nm1Gen(Element *rankingMatrix) {

    // make tuple of ints for pairs (lvalue, column)
    thrust::pair<int, thrust::pair<int, int> > *nm1GenPairs;
    cudaMallocManaged(&nm1GenPairs, sizeof(nm1GenPairs[0]) * (n * n));

    protoNM1Gen<<<n, n>>>(rankingMatrix, nm1GenPairs);
    cudaDeviceSynchronize();

    nm1GenDevice<<<n, n/2>>>(rankingMatrix, nm1GenPairs, n);
    cudaDeviceSynchronize();

    cudaFree(nm1GenPairs);
}

// function to mark the nm_1 pairs in the ranking matrix
void nm1(Element *rankingMatrix, bool *nm1Pairs) {

    // initialize nm1Pairs array entries to false
    nm1False<<<2, n>>>(nm1Pairs);

    cudaDeviceSynchronize();

    // make tuple of ints for pairs (lvalue, column)
    thrust::pair<int, thrust::pair<int, int> > *localNM1Pairs;

    cudaMallocManaged(&localNM1Pairs, sizeof(localNM1Pairs[0]) * (n * n));

    protoNM1<<<n, n>>>(rankingMatrix, localNM1Pairs);

    cudaDeviceSynchronize();

    nm1Device<<<n, n/2>>>(rankingMatrix, localNM1Pairs, nm1Pairs, n);
    cudaDeviceSynchronize();

    cudaFree(localNM1Pairs);

}

// function to mark the nm_2-generating pairs in the ranking matrix
void nm2GenHost(Element *rankingMatrix,
                thrust::pair<int,int> *matchingPairs,
                bool *nm1Pairs,
                int n) {

    Element **nm2GenPairPointers;

    cudaMallocManaged(&nm2GenPairPointers, 2 * n * sizeof(*nm2GenPairPointers));

    nm2GenDevice<<<n, n>>>(rankingMatrix, matchingPairs, nm2GenPairPointers);

    cudaDeviceSynchronize();

    nm2GenPointers<<<n, n>>>(rankingMatrix, matchingPairs, nm1Pairs, nm2GenPairPointers);

    cudaDeviceSynchronize();

    cudaFree(nm2GenPairPointers);
}

#define BILLION 1000000000L

// main function
int main(int argc, char **argv) {

    srand(time(NULL));

    n = 0; // start with no participants

    createSignalHandlers(); // for stopping algorithm

    checkArgs(argc);

    // make a pointer to the path of the file from the CLI arguments
    const char *UsersFile;
    UsersFile = argv[1];

    // make a file stream called inFile from the users file
    ifstream inFile(UsersFile,ifstream::in);

    // make string to hold file information
    string s = "";

    // if the file exists and is open
    if (inFile.is_open()) {

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

    // make a string stream from string read from file
    stringstream ss(s);

    // divide n by 2 because we have menPrefs and womenPrefs in the same file
    n = n / 2;

    // create element pointer for Ranking Matrix
    Element *rankingMatrix;

    // allocate memory on GPU for rankingMatrix
    cudaMallocManaged(&rankingMatrix, (sizeof(*rankingMatrix) * (n * n)));

    // create ranking matrix
    // add mens prefs to the matrix by row
    // add x and y values at the same time
    // matrix is read left -> right, top -> down
    // this is generation of first arbitrary matching
    for (int i = 0, col = 0, row = 1; i < n * n; i++) {
        // initialize r Value, type and pointer
        // set r value to default to 0
        rankingMatrix[i].r = 0;

        rankingMatrix[i].type = 0; // mark unmatched
        rankingMatrix[i].nm2gen = false;
        // point to thyself
        rankingMatrix[i].pointer = &rankingMatrix[i];

        // set the l value based on Mans preference
        ss >> rankingMatrix[i].l;

        // if i%n is 0 we are at the next Y position
        if (0 == i % n) {

            // increment col for a new col
            col++;
        }

        // for each col set y to that col
        rankingMatrix[i].x = col;

        // set the x value
        rankingMatrix[i].y = row;

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
        ss >> rankingMatrix[(i * n) + colsDone].r;

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

    // allocate stable
    cudaMallocManaged(&stable, sizeof(*stable));

    // allocate pairs on the GPU, n for forward, n for reverse
    cudaMallocManaged(&matchingPairs, sizeof(matchingPairs[0]) * 2 * n);
    // timing Code
    uint64_t diff;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC,&start);
    do{
        // create initial matching
        initMatch(rankingMatrix);

        for(int c = 0; c < n; c++){
            *stable = true;

            // check for unstable pairs
            unstable<<<n, n>>>(rankingMatrix, matchingPairs, stable);

            cudaDeviceSynchronize();

            if(*stable){

                clock_gettime(CLOCK_MONOTONIC, &end);
                diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
                printPairs();
                break;
            }

            nm1Gen(rankingMatrix);

            bool *nm1Pairs;
            cudaMallocManaged(&nm1Pairs, 2 * n * sizeof(*nm1Pairs));
            nm1(rankingMatrix, nm1Pairs);

            // mark the nm_2-generating pairs
            nm2GenHost(rankingMatrix, matchingPairs, nm1Pairs, n);

            cudaFree(nm1Pairs);

            nm2Pairs<<<n, n>>>(rankingMatrix);

            cudaDeviceSynchronize();

            newMatching<<<n, n>>>(rankingMatrix, matchingPairs);

            cudaDeviceSynchronize();

            resetAfterIteration<<<n, n>>>(rankingMatrix);
            cudaDeviceSynchronize();

        }// end of for
        resetAll<<<n, n>>>(rankingMatrix, matchingPairs);

        cudaDeviceSynchronize();
    }
    while(!*stable);// end of while

    cudaFree(matchingPairs);

    cudaFree(stable);

    cudaFree(rankingMatrix);

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

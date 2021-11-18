#include <iostream>
#include <chrono>
using namespace std;

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    int times = atoi(argv[2]);
    cout << N << "*" << N << ", " << times << "times" << endl;

    float A[N*N], B[N*N], C[N*N];
    for(int j=0;j<N;j++){
    	for(int i=0;i<N;i++){
	   A[i + j*N] = i;
	   B[i + j*N] = j;
	   C[i + j*N] = 0.0;
	}
    }

    auto start = chrono::system_clock::now();

    for(int j=0;j<N;j++){
    	for(int i=0;i<N;i++){
           for(int k=0;k<1024;k++)
                C[i + j*N] += A[i + j*N]*3.14 + B[i + j*N]/3.14;
	}
    }
    

    auto end = chrono::system_clock::now();
    auto dur = end - start;
    cerr << (double)(chrono::duration_cast<chrono::nanoseconds>(dur).count())/1000000 << "msec" << endl;

    cout << "C" << endl;
    for(int j=N-1;j<N;j++){
    	for(int i=0;i<N;i++){
		cout << C[i + j*N] << ' ';
	}
	cout << endl;
    }

    return 0;
}


/* reference */ 
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// https://qiita.com/wazakkyd/items/8a5694e7a001465b6025

#include <iostream>
#include <chrono>
using namespace std;

__global__ void MatAdd(float *A, float *B, float *C, int times)
{
    int block_idx = blockIdx.x*blockDim.x*blockDim.y;
    int i = threadIdx.x + blockDim.x * threadIdx.y + block_idx;
    for(int k=0;k<times;k++)
        C[i] += A[i]*3.14 + B[i]/3.14;
}

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

    float *a, *b, *c;
    cudaMalloc(&a, N*N*sizeof(float));
    cudaMalloc(&b, N*N*sizeof(float));
    cudaMalloc(&c, N*N*sizeof(float));

    cudaMemcpy(a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, N*N*sizeof(float), cudaMemcpyHostToDevice);

    MatAdd<<<N, N>>>(a, b, c, times);

    cudaMemcpy(C, c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
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


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}


/* reference */ 
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// https://qiita.com/wazakkyd/items/8a5694e7a001465b6025

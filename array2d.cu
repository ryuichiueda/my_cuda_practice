#include <iostream>
#include <chrono>

__global__ void MatAdd(float *A, float *B, float *C)
{
    int block_idx = blockIdx.x*blockDim.x*blockDim.y;
    int i = threadIdx.x + blockDim.x * threadIdx.y + block_idx;
    for(int k=0;k<1024;k++)
        C[i] += A[i]*3.14 + B[i]/3.14;
}

int main(int argc, char **argv)
{
    int N = atoi(argv[1]);
    std::cout << N << std::endl;

    float A[N*N], B[N*N], C[N*N];
    for(int j=0;j<N;j++){
    	for(int i=0;i<N;i++){
	   A[i + j*N] = i;
	   B[i + j*N] = j;
	   C[i + j*N] = 0.0;
	}
    }

    auto start = std::chrono::system_clock::now();

    float *a, *b, *c;
    cudaMalloc(&a, N*N*sizeof(float));
    cudaMalloc(&b, N*N*sizeof(float));
    cudaMalloc(&c, N*N*sizeof(float));

    cudaMemcpy(a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, N*N*sizeof(float), cudaMemcpyHostToDevice);

    MatAdd<<<N, N>>>(a, b, c);

    cudaMemcpy(C, c, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    auto end = std::chrono::system_clock::now();
    auto dur = end - start;
    std::cerr << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count() << "msec" << std::endl;

    std::cout << "C" << std::endl;
    for(int j=N-1;j<N;j++){
    	for(int i=0;i<N;i++){
		std::cout << C[i + j*N] << ' ';
	}
	std::cout << std::endl;
    }


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}


/* reference */ 
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// https://qiita.com/wazakkyd/items/8a5694e7a001465b6025

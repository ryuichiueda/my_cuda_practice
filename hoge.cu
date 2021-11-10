#include <iostream>

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    float A[] = {1,2,3};
    float B[] = {2,3,4};
    float C[] = {0,0,0};

    float *a, *b, *c;
    cudaMalloc(&a, 3*sizeof(float));
    cudaMalloc(&b, 3*sizeof(float));
    cudaMalloc(&c, 3*sizeof(float));

    cudaMemcpy(a, A, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, 3*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    VecAdd<<<1, 3>>>(a, b, c);

    cudaMemcpy(C, c, 3*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << C[0] << " " << C[1] << " " << C[2] << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}


/* reference */ 
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// https://qiita.com/wazakkyd/items/8a5694e7a001465b6025

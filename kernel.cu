#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip> // For std::setw

#include "helper.h"

#define TYPE __half
#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)
#define TOFLOAT(val) CAT(TYPE, 2float)(val)

using namespace nvcuda;
constexpr int n = 1 << 4;

Matrix<TYPE> A, B, C;
Matrix<TYPE> A_gpu, B_gpu, C_gpu;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <typename T, typename oT>
__global__ void matrixMultiply(Matrix<T> A, Matrix<T> B, Matrix<T> C)
{

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, oT> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, A.data, 16);
    wmma::load_matrix_sync(b_frag, B.data, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(C.data, c_frag, 16, wmma::mem_row_major);
}

void printMatrix(const char& name, Matrix<TYPE>& mat) {
    std::cout << "Matrix " << name << " :" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << TOFLOAT(mat.data[i * n + j]) << " ";
        }
        std::cout << std::endl;
    }
}

void initMatricies()
{
    initMatrixCPU<TYPE>(A, n);
    initMatrixCPU<TYPE>(B, n);
    initMatrixCPU<TYPE>(C, n);

    printMatrix('A', A);
    printMatrix('B', B);

    
    allocateMatrixOnGPU<TYPE> (A_gpu, n);
    allocateMatrixOnGPU<TYPE> (B_gpu, n);
    allocateMatrixOnGPU<TYPE>(C_gpu, n);

    copyMatrixToGPU<TYPE>(A, A_gpu);
    copyMatrixToGPU<TYPE>(B, B_gpu);
}

void freeMatricies()
{
    copyMatrixToCPU<TYPE>(C_gpu, C);

    printMatrix('C', C);
    testMatrix<TYPE>(A, B, C);

    freeMatrixOnCPU<TYPE>(A);
    freeMatrixOnCPU<TYPE>(B);
    freeMatrixOnCPU<TYPE>(C);

    freeMatrixOnGPU<TYPE>(A_gpu);
    freeMatrixOnGPU<TYPE>(B_gpu);
    freeMatrixOnGPU<TYPE>(C_gpu);
}

void dispath(const unsigned int& threads)
{
    int blocks = (n + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    matrixMultiply<__half, __half><<< 1, 32 >>>(A_gpu, B_gpu, C_gpu);
}

int main()
{
    initMatricies();
    dispath(16);
    freeMatricies();
    return 0;
}

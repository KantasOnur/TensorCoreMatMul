#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip> // For std::setw

#include "helper.h"

#define TYPE __half
#define OUTTYPE float


using namespace nvcuda;
constexpr int n = 1 << 4;

Matrix<TYPE> A, B;
Matrix<TYPE> A_gpu, B_gpu;

Matrix<OUTTYPE> C, C_gpu;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;


template <typename T, typename To>
__global__ void matrixMultiply(Matrix<T> A, Matrix<T> B, Matrix<To> C)
{
    // Each block computes one tile (WMMA_M x WMMA_N) of the output matrix C.
    unsigned int warpX = blockIdx.x;
    unsigned int warpY = blockIdx.y;
    unsigned int n = A.size;

    // Declare the WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, To> c_frag;

    // Initialize the output fragment to zero
    wmma::fill_fragment(c_frag, (To)0.0);

    int cRow = warpY * WMMA_M;
    int cCol = warpX * WMMA_N;

    for (int k = 0; k < n; k += WMMA_K) {

        int aRow = warpY * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpX * WMMA_N;

        // Load the input fragments from global memory
        wmma::load_matrix_sync(a_frag, &A.data[aRow * n + aCol], n);
        wmma::load_matrix_sync(b_frag, &B.data[bRow * n + bCol], n);

        // Perform the matrix multiplication on the fragment tiles
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output tile in C
    wmma::store_matrix_sync(&C.data[cRow * n + cCol], c_frag, n, wmma::mem_row_major);
}

void printMatrix(const char& name, Matrix<TYPE>& mat) {
    std::cout << "Matrix " << name << " :" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << (float) mat.data[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

void initMatricies()
{
    initMatrixCPU<TYPE>(A, n);
    initMatrixCPU<TYPE>(B, n);
    initMatrixCPU<OUTTYPE>(C, n);

    //printMatrix('A', A);
    //printMatrix('B', B);

    
    allocateMatrixOnGPU<TYPE> (A_gpu, n);
    allocateMatrixOnGPU<TYPE> (B_gpu, n);
    allocateMatrixOnGPU<OUTTYPE>(C_gpu, n);

    copyMatrixToGPU<TYPE>(A, A_gpu);
    copyMatrixToGPU<TYPE>(B, B_gpu);
}

void freeMatricies()
{
    copyMatrixToCPU<OUTTYPE>(C_gpu, C);

    //printMatrix('C', C);
    testMatrix<TYPE, OUTTYPE>(A, B, C);

    freeMatrixOnCPU<TYPE>(A);
    freeMatrixOnCPU<TYPE>(B);
    freeMatrixOnCPU<OUTTYPE>(C);

    freeMatrixOnGPU<TYPE>(A_gpu);
    freeMatrixOnGPU<TYPE>(B_gpu);
    freeMatrixOnGPU<OUTTYPE>(C_gpu);
}

void dispath(const unsigned int& threads)
{
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(n / 16, n / 16);
    matrixMultiply<TYPE, OUTTYPE><<< BLOCKS, 32 >>>(A_gpu, B_gpu, C_gpu);
    std::cout << "finished matmul" << std::endl;
}

int main()
{
    initMatricies();
    std::cout << "discpatched" << std::endl;
    dispath(16);
    freeMatricies();
    return 0;
}

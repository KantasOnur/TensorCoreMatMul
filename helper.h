#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <vector>
#include <cassert>

template <typename T>
struct Matrix
{
    int size;
    T* data;
};

template < typename T>
void allocateMatrixOnGPU(Matrix<T>& mat, int size)
{
    mat.size = size;
    cudaMalloc(&mat.data, mat.size * mat.size * sizeof(T));
}

template < typename T>
void freeMatrixOnGPU(Matrix<T>& mat)
{
    cudaFree(mat.data);
}

template <typename T>
void initMatrixCPU(Matrix<T>& mat, int size)
{
    mat.size = size;
    mat.data = new T[mat.size * mat.size];
    for (int i = 0; i < mat.size * mat.size; i++) {
        mat.data[i] = static_cast<T>((rand() % 10000) / 1000.0f); // Random float values between 0 and 9.999
    }
}

template <typename T>
void copyMatrixToGPU(const Matrix<T>& cpuMatrix, Matrix<T>& gpuMatrix) {
    cudaMemcpy(gpuMatrix.data, cpuMatrix.data,
        cpuMatrix.size * cpuMatrix.size * sizeof(T),
        cudaMemcpyHostToDevice);
}

template <typename T>
void copyMatrixToCPU(const Matrix<T>& gpuMatrix, Matrix<T>& cpuMatrix) {
    cudaMemcpy(cpuMatrix.data, gpuMatrix.data,
        gpuMatrix.size * gpuMatrix.size * sizeof(T),
        cudaMemcpyDeviceToHost);
}

template <typename T>
void freeMatrixOnCPU(Matrix<T>& mat) {
    delete[] mat.data;
}

template <typename T, typename To>
void testMatrix(Matrix<T>& A, Matrix<T>& B, Matrix<To>& C) {
    
    int n = A.size;
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            To sum = 0;
            for (int k = 0; k < n; ++k)
            {
                sum += (To)A.data[row * n + k] * (To)B.data[k * n + col];
            }
            std::cout << "expected: " << (float)sum << " recieved: " << (float)C.data[row * n + col] << std::endl;
        }
    }
}

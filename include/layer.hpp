#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

struct Matrix
{
    int height;
    int width;
    float *elements;
};

struct NeuralNetwork {
    Matrix W1, b1, W2, b2;
    Matrix Z1, A1, Z2, Probs;
    Matrix dW1, db1, dW2, db2;
    Matrix dZ1, dZ2;
    Matrix Y_one_hot;
};

void gpu_matmul(Matrix A, Matrix B, Matrix bias, Matrix C);
void gpu_mat_sub(float* A, float* B, float* C, int N);
void gpu_relu_forward(float* a, float* b, float alpha, int N);
void gpu_relu_backward(float* dz, float* z_fwd, int N);
void gpu_softmax(float* input, float* output, int num_rows);
void gpu_one_hot(int* labels, float* output, int num_rows);
void gpu_update_params(float* param, float* grads, float lr, int N);
void gpu_calculate_error(float* A2, float* Y, float* dZ2, int N);
void allocate_network_gpu(NeuralNetwork& d_net, const NeuralNetwork& h_net);
void gpu_matmul_AT_B(Matrix A, Matrix B, Matrix C, float batch_size);
void gpu_matmul_A_BT(Matrix A, Matrix B, Matrix C);
void gpu_bias_grad(float* dZ, float* db, int rows, int cols, float batch_size);

void init_params(Matrix& m, float n_in);
void init_bias(Matrix& b);
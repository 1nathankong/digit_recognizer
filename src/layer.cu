#include "data_loader.hpp"
#include <random>
#include <cmath>
#include <cstdio>
// ---Global Variables---
#define NUM_NEURONS 10
#define TILE_WIDTH 16
#include "layer.hpp"
/*
---------------------------MatMul--------------------------------------------------------------
*/
__global__ void matMul_kernel(Matrix A, Matrix B, Matrix bias, Matrix C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float cValue = 0.0f;

    if(row < A.height && col < B.width) {
        for (int i = 0; i < A.width; i++)
        {
            cValue += A.elements[row * A.width + i] * B.elements[i * B.width + col];
        }
        cValue += bias.elements[col];
        C.elements[row * C.width + col] = cValue;
    }
}

/*
__global__ void tiled_matMul_kernel(Matrix A, Matrix B, Matrix bias, Matrix C)
{
    __shared__ shareA[TILE_WIDTH][TILE_WIDTH];
    __shared__ shareB[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float cValue = 0.0f;
    for(int m = 0; m < (A.width + TILE_WIDTH - 1)/ TILE_WIDTH; ++m)
    {
        if(row < A.height && (m * TILE_WIDTH + tx) < A.width) {
            shareA[tx][ty] = A.elements[row*A.width + (m * TILE_WIDTH + tx)];
        }
        else {
            shareA[tx][ty] = 0.0f;
        }
        if (col < B.width && (m * TILE_WIDTH + ty) < B.width) {
            shareB[ty][tx] = B.elements[(m * TILE_WIDTH + ty) * B.width + col];
        }
        else {
            shareB[ty][tx] = 0.0f;
        }
        
        __syncthreads();


        for (int i = 0; i < TILE_WIDTH; ++i)
        {
            cValue += shareA[ty][k] * shareB[k][tx];
        }

        __syncthreads();
    }
    if(row < A.height && col < B.width) {
        cValue += bias.elements[col];
        C.elements[row * C.width + col] = cValue;
    }
    
}
*/

__global__ void matMul_AT_B_kernel(Matrix A, Matrix B, Matrix C, float batch_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < A.width && col < B.width) {
        float sum = 0.0f;
        for (int i = 0; i < A.height; i++) {
            sum += A.elements[i * A.width + row] *   // A transposed
                   B.elements[i * B.width + col];
        }
        C.elements[row * C.width + col] = sum / batch_size;
    }
}

__global__ void bias_grad_kernel(float* dZ, float* db, int rows, int cols, float batch_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            sum += dZ[r * cols + col];
        }
        db[col] = sum / batch_size;
    }
}

__global__ void matmul_A_BT_kernel(Matrix A, Matrix B, Matrix C)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < A.height && col < B.height) {
        float sum = 0.0f;
        for (int i = 0; i < A.width; i++) {
            sum += A.elements[row * A.width + i] *
                   B.elements[col * B.width + i];  // B transposed
        }
        C.elements[row * C.width + col] = sum;
    }
}


__global__ void mat_sub_kernel(float* A, float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] - B[i]; 
    }
}

/*
------------------------Kernel Logic--------------------------------------------------------------------------------------------
*/

//ReLU forward kernel 
__global__ void relu_forward_kernel(const float* d_a, float* d_b, float alpha, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        d_b[col] = fmaxf(alpha * d_a[col], d_a[col]);
    }
    
}

//ReLU backward kernel 
__global__ void relu_backward_kernel(float* d_z, float* z_forward, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (z_forward[i] <= 0.0f) {
        d_z[i] = 0.0f;
    }
}

//softmax kernel (optimized for 10 neurons in output layer)
__global__ void softmax_kernel(float* input, float* output, int num_rows)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= num_rows) return;
    if(tid >= NUM_NEURONS) return;
    
    __shared__ float s_data[NUM_NEURONS]; // shared memory 
    
    float val = input[row * NUM_NEURONS + tid]; 

    s_data[tid] = val;
    __syncthreads();

    float max_val = s_data[0];

    for(int i = 0; i < NUM_NEURONS; ++i) //find max value across the neurons
    {
        if (s_data[i] > max_val) {
            max_val = s_data[i];
        }
    }

    //compute exponent
    float exp_val = expf(val - max_val);
    s_data[tid] = exp_val;
    __syncthreads();

    float sum_exp = 0.0;

    for (int i = 0; i < NUM_NEURONS; ++i)
    {
        sum_exp += s_data[i];
    }

    output[row * NUM_NEURONS + tid] = exp_val / sum_exp;
}   

//implement one hot
__global__ void one_hot_kernel(int* labels, float* output, int num_rows)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= num_rows) return;
    if(tid >= NUM_NEURONS) return;

    int target_label = labels[row];

    if (tid == labels[row]) {   
        output[row * NUM_NEURONS + tid] = 1.0f;
    }
    else {
        output[row * NUM_NEURONS + tid] = 0.0f;
    }

}

__global__ void update_param_kernel(float* param, const float* grads, float alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        param[i] -= alpha * grads[i];
    }
}

/*
---Launch Configurations for main.cpp file----
*/
void gpu_matmul(Matrix A, Matrix B, Matrix bias, Matrix C) {
    // 2D grid for matrix
    dim3 block(16, 16);
    dim3 grid((C.height + 15) / 16, (C.width + 15) / 16);
    matMul_kernel<<<grid, block>>>(A, B, bias, C);
}

void gpu_relu_forward(float* a, float* b, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(a, b, alpha, N);
}

void gpu_relu_backward(float* d_z, float* z_forward, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    relu_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_z, z_forward, N);
}

void gpu_softmax(float* input, float* output, int num_rows) {
    // Each block handles one row (one image)
    // threads = 10 (NUM_NEURONS)
    softmax_kernel<<<num_rows, 10>>>(input, output, num_rows);
}

void gpu_one_hot(int* labels, float* output, int num_rows) {
    one_hot_kernel<<<num_rows, 10>>>(labels, output, num_rows);
}

void gpu_update_params(float* param, float* grads, float lr, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_param_kernel<<<blocks, threads>>>(param, grads, lr, N);
}

void gpu_matmul_AT_B(Matrix A, Matrix B, Matrix C, float batch_size) {
    dim3 block(16, 16);
    dim3 grid((C.height + 15) / 16, (C.width + 15) / 16);
    matMul_AT_B_kernel<<<grid, block>>>(A, B, C, batch_size);
}

void gpu_matmul_A_BT(Matrix A, Matrix B, Matrix C) {
    dim3 block(16, 16);
    dim3 grid((C.height + 15) / 16, (C.width + 15) / 16);
    matmul_A_BT_kernel<<<grid, block>>>(A, B, C);
}

void gpu_bias_grad(float* dZ, float* db, int rows, int cols, float batch_size) {
    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    bias_grad_kernel<<<blocks, threads>>>(dZ, db, rows, cols, batch_size);
}

/*
--- Other intializations of variables ---
*/

void init_params(Matrix& m, float n_in) 
{
    std::default_random_engine gen;
    float stddev = std::sqrt(2.0f / n_in);
    std::normal_distribution<float> dist(0.0f, stddev);

    for(int i = 0; i < m.height * m.width; ++i)
    {
        m.elements[i] = dist(gen);
    }
}

void init_bias(Matrix& b)
{
    for(int i = 0; i < b.height * b.width; ++i)
    {
        b.elements[i] = 0.0f;
    }
}

/*
---Error Calculation
*/
void gpu_calculate_error(float* A2, float* Y, float* dZ2, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    // Uses your existing mat_sub_kernel
    mat_sub_kernel<<<blocks, threads>>>(A2, Y, dZ2, N);
}

/*
---CPU to GPU allocation---
*/
void allocate_network_gpu(NeuralNetwork& d_net, const NeuralNetwork& h_net)
{
    d_net.W1 = {h_net.W1.height, h_net.W1.width, nullptr};
    cudaMalloc(&d_net.W1.elements, d_net.W1.height * d_net.W1.width * sizeof(float));
    cudaMemcpy(d_net.W1.elements, h_net.W1.elements, d_net.W1.height * d_net.W1.width * sizeof(float), cudaMemcpyHostToDevice);

    d_net.b1 = {h_net.b1.height, h_net.b1.width, nullptr};
    cudaMalloc(&d_net.b1.elements, d_net.b1.height * d_net.b1.width * sizeof(float));
    cudaMemcpy(d_net.b1.elements, h_net.b1.elements, d_net.b1.height * d_net.b1.width * sizeof(float), cudaMemcpyHostToDevice);

    d_net.W2 = {h_net.W2.height, h_net.W2.width, nullptr};
    cudaMalloc(&d_net.W2.elements, d_net.W2.height * d_net.W2.width * sizeof(float));
    cudaMemcpy(d_net.W2.elements, h_net.W2.elements, d_net.W2.height * d_net.W2.width * sizeof(float), cudaMemcpyHostToDevice);

    d_net.b2 = {h_net.b2.height, h_net.b2.width, nullptr};
    cudaMalloc(&d_net.b2.elements, d_net.b2.height * d_net.b2.width * sizeof(float));
    cudaMemcpy(d_net.b2.elements, h_net.b2.elements, d_net.b2.height * d_net.b2.width * sizeof(float), cudaMemcpyHostToDevice);

    d_net.Z1 = {64,512, nullptr};
    cudaMalloc(&d_net.Z1.elements, 64 * 512 * sizeof(float));

    d_net.A1 = {64, 512, nullptr};
    cudaMalloc(&d_net.A1.elements, 64 * 512 * sizeof(float));

    d_net.Probs = {64, 10, nullptr};
    cudaMalloc(&d_net.Probs.elements, 64 * 10 * sizeof(float));

    d_net.Z2 = {64, 10, nullptr};
    cudaMalloc(&d_net.Z2.elements, 64 * 10 * sizeof(float));

    d_net.dZ2 = {64, 10, nullptr};
    cudaMalloc(&d_net.dZ2.elements, 64 * 10 * sizeof(float));

    d_net.dZ1 = {64, 512, nullptr};
    cudaMalloc(&d_net.dZ1.elements, 64 * 512 * sizeof(float));

    d_net.dW2 = {512, 10, nullptr};
    cudaMalloc(&d_net.dW2.elements, 512 * 10 * sizeof(float));

    d_net.dW1 = {784, 512, nullptr};
    cudaMalloc(&d_net.dW1.elements, 784 * 512 * sizeof(float));

    d_net.db2 = {1, 10, nullptr};
    cudaMalloc(&d_net.db2.elements, 10 * sizeof(float));

    d_net.db1 = {1, 512, nullptr};
    cudaMalloc(&d_net.db1.elements, 512 * sizeof(float));

    d_net.Y_one_hot = {64, 10, nullptr};
    cudaMalloc(&d_net.Y_one_hot.elements, 64 * 10 * sizeof(float));
}





// ---Global Variables---
#define NUM_NEURONS 10

/*
---------------------------Matrix Struct & MatMul--------------------------------------------------------------
*/

struct Matrix
{
    int height;
    int width;
    float *elements;
};

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

    for(int i = 0; i < NUM_NEURONS; i++) //find max value across the neurons
    {
        if (s_data[i] > max_val) {
            max_val = s_data[i];
        }
    }

    //compute exponent
    float exp_val = expf(val - max_val);
    s_data[tid] = val;
    __syncthreads();

    float sum_exp = 0.0;

    for (int i = 0; i < NUM_NEURONS; i++)
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


#include "data_loader.hpp"
#include "layer.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cuda_runtime.h>
#include <chrono>

int main() {
    // Force CUDA init
    cudaSetDevice(0);

    // Load data on CPU
    Dataset data;
    load("../", data);
    std::cout << "Data loaded. Images: " << data.images.size() 
              << " Labels: " << data.labels.size() << std::endl;

    // Initialize CPU network
    NeuralNetwork h_net;
    h_net.W1 = {784, 512, new float[784 * 512]};
    h_net.b1 = {1, 512, new float[512]};
    h_net.W2 = {512, 10, new float[512 * 10]};
    h_net.b2 = {1, 10, new float[10]};

    init_params(h_net.W1, 784.0f);
    init_bias(h_net.b1);
    init_params(h_net.W2, 512.0f);
    init_bias(h_net.b2);

    // Allocate GPU network
    NeuralNetwork d_net;
    allocate_network_gpu(d_net, h_net);
    std::cout << "GPU network allocated." << std::endl;

    // Allocate GPU batch buffers
    float* d_batch_images;
    int* d_batch_labels;
    cudaMalloc(&d_batch_images, 64 * 784 * sizeof(float));
    cudaMalloc(&d_batch_labels, 64 * sizeof(int));

    // Setup batch matrix
    Matrix d_batch_images_matrix;
    d_batch_images_matrix.height = 64;
    d_batch_images_matrix.width = 784;
    d_batch_images_matrix.elements = d_batch_images;

    // CPU batch buffers
    float h_batch_images[64 * 784];
    int h_batch_labels[64];

    // Training setup
    std::vector<int> indices(60000);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(std::random_device{}());
    float alpha = 0.1f;
    std::cout << "Starting training..." << std::endl;
    cudaDeviceSynchronize();
    auto total_start = std::chrono::high_resolution_clock::now();
    for(int epoch = 0; epoch < 6; ++epoch)
    {
        int correct = 0;
        float loss = 0.0f;
        //cudaDeviceSynchronize();
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), g);

        for(int i = 0; i < 60000; i += 64)
        {
            int batch_size = std::min(64, 60000 - i);

            // Assemble batch on CPU
            for(int b = 0; b < batch_size; ++b)
            {
                int idx = indices[i + b];
                memcpy(h_batch_images + b * 784, 
                       &data.images[idx * 784], 
                       784 * sizeof(float));
                h_batch_labels[b] = data.labels[idx];
            }

            // Single transfer to GPU
            cudaMemcpy(d_batch_images, h_batch_images, 
                batch_size * 784 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_batch_labels, h_batch_labels,
                batch_size * sizeof(int), cudaMemcpyHostToDevice);

            // Forward pass
            gpu_matmul(d_batch_images_matrix, d_net.W1, d_net.b1, d_net.Z1);
            gpu_relu_forward(d_net.Z1.elements, 
                d_net.A1.elements, 0.01f, 64 * 512);
            gpu_matmul(d_net.A1, d_net.W2, d_net.b2, d_net.Z2);
            gpu_softmax(d_net.Z2.elements, d_net.Probs.elements, 64);

            // Backward pass
            gpu_one_hot(d_batch_labels, d_net.Y_one_hot.elements, 64);
            gpu_calculate_error(d_net.Probs.elements, 
                d_net.Y_one_hot.elements, d_net.dZ2.elements, 64 * 10);
            gpu_matmul_AT_B(d_net.A1, d_net.dZ2, d_net.dW2, 64.0f);
            gpu_bias_grad(d_net.dZ2.elements, 
                d_net.db2.elements, 64, 10, 64.0f);
            gpu_matmul_A_BT(d_net.dZ2, d_net.W2, d_net.dZ1);
            gpu_relu_backward(d_net.dZ1.elements, 
                d_net.Z1.elements, 64 * 512);
            gpu_matmul_AT_B(d_batch_images_matrix, d_net.dZ1, d_net.dW1, 64.0f);
            gpu_bias_grad(d_net.dZ1.elements, 
                d_net.db1.elements, 64, 512, 64.0f);

            // Update parameters
            gpu_update_params(d_net.W2.elements, 
                d_net.dW2.elements, alpha, 512 * 10);
            gpu_update_params(d_net.b2.elements, d_net.db2.elements, alpha, 10);
            gpu_update_params(d_net.W1.elements, 
                d_net.dW1.elements, alpha, 784 * 512);
            gpu_update_params(d_net.b1.elements, 
                d_net.db1.elements, alpha, 512);
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_seconds = std::chrono::duration<float>(epoch_end - epoch_start).count();
        float throughput = 60000.0f / epoch_seconds;

        // Epoch loss
        float h_probs[64 * 10];
        cudaMemcpy(h_probs, d_net.Probs.elements, 
            64 * 10 * sizeof(float), cudaMemcpyDeviceToHost);
        
        
        for(int b = 0; b < 64; ++b) 
        {
            int label = h_batch_labels[b];
            if(label >= 0 && label < 10) {
                loss -= logf(h_probs[b * 10 + label] + 1e-8f);
            }
            int predicted = 0;
            float max_prob = h_probs[b*10];
            for(int c = 1; c < 10; ++c) {
                if(h_probs[b * 10 + c] > max_prob) {
                    max_prob = h_probs[b * 10 + c];
                    predicted = c;
                }
            }
            if(predicted == label) correct++;
        }
        cudaDeviceSynchronize();
        loss /= 64.0f;
        float accuracy = (float)correct / 64.0f * 100.0f;
        std::cout << "Epoch " << epoch << " | alpha: " << alpha
        << " | loss: " << loss
        << " | accuracy: " << accuracy << "%"
        << " | Throughput: " << throughput << " images/sec"
        << std::endl;
    }
    

    // Free GPU memory
    cudaFree(d_batch_images);
    cudaFree(d_batch_labels);
    cudaFree(d_net.W1.elements);
    cudaFree(d_net.W2.elements);
    cudaFree(d_net.b1.elements);
    cudaFree(d_net.b2.elements);
    cudaFree(d_net.Z1.elements);
    cudaFree(d_net.A1.elements);
    cudaFree(d_net.Z2.elements);
    cudaFree(d_net.Probs.elements);
    cudaFree(d_net.dZ1.elements);
    cudaFree(d_net.dZ2.elements);
    cudaFree(d_net.dW1.elements);
    cudaFree(d_net.dW2.elements);
    cudaFree(d_net.db1.elements);
    cudaFree(d_net.db2.elements);
    cudaFree(d_net.Y_one_hot.elements);

    // Free CPU memory
    delete[] h_net.W1.elements;
    delete[] h_net.W2.elements;
    delete[] h_net.b1.elements;
    delete[] h_net.b2.elements;

    return 0;
}
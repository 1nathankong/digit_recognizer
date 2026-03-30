#pragma once 

namespace neural_network {

    // none for input layer, hidden layers use ReLU, and output layer uses Softmax
    enum class Activation {ReLU, Softmax, None}; 

    class Layer {
        public: 
            int in_size, out_size; // M,N Dimensions
            Activation activation;
            //pointers that GPU uses
            float *d_weights, *d_biases; 
            float *d_dW, *d_dB;
            float *d_Z, *d_A;

            // allocation gpu memory and intialize weights
            Layer(int in, int out, Activation act = Activation::ReLU);
            //clear up gpu memory
            ~Layer();
            // forward pass to call kernel
            //void feed_forward(float *d_input, int m);
            //backprop to update weights using gradients
            //void back_propagation(float *d_input, float *d_next_Z, int m);
            
    };
}
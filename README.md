Learning how to code without Generative AI and learn how to do Mathmatical modeling step by step. This example is an implementation of a Feedforward Neural Network using Backpropagation and Gradient Descent. 


Dataset I used: MNIST numbers, csv and binary(idx)

Update status:

First got basic Neural Network with backprop working with 95% accuracy in 3000 iterations, iterations became tedious so I began training in epochs/batches to see how my model would react with more stability and faster training. 

Rewrote my Gradient Descent Function as a SGD so I can look at 1 small subset iteration when training. 

C++ rewrite
    
    /include
        data_loader.hpp - Load the MNIST dataset in C++ (done)
        layer.hpp - weights and bias declared and also declaration of compuations done with GPU (done)

    /src
        data_loader.cpp - read idx data from data_loader and format to use for Neural Network. Load onto GPU then copy all at once to GPU. (done)
        layer.cu - raw computations done on GPU (done)
        main.cu - all files get called here and executes. allocate and deallocate pointers and free memory after run ends (done)
        

    CMakeLists.txt - compiler code to setup model in C++ (done)

Summary of optimizations:
- When loading the idx data set I used the C++23 libary to flip the bits from big endian to little endian, it is faster to use dedicated instructions that is baked into the x86 assembly than doing manual bit manipulation. One thing to note it is able to do the reverse byte order in a single clock cycle. 
- Rewrote kernels from scratch, did some hardcodings based on how the layers and neurons were set up. I wrote each kernel operation and had a function for every possible step needed for neural network
- For instance had functions to perform matrix multiplcations for layers that needed transposes, or if for the output layer since I knew it was going to be 1 dimensional, I flatted to be column major since memory access is designed that way for GPU. 
- My orignal implementations were feature first, but to process data in parallel and maximize efficiency, updating my model to become batch first.
- Hardcoding neurons in output layer, rather than looking through all the threads, i only would look at the first 10 threads and only the first 10 threads, so it reduces the time loading 10 values. The important fact to get out of it is that I avoid thread synchronization or any other value comparisons in any registers that are not used. 

Graphed data of Results:

I got my data from my C++ model and compared it to the Pytorch model. It seems all my performance optimizations really paid off when comparing to Pytorch. 

Graph 1 of Kernel Operation in raw %
![alt text](benchmark_results.png)

Graph 2 Kernel Operation Multiplier values
![alt text](performance_multiplier.png)

Reflection:

Accuracy 97% in the sgd variant models. 

C++ rewrite has 3.51x more throughput than Python Version with no Pytorch. C++ can process 132,131 images/sec, while Python ranges between 20,000-30,000 images/sec
C++ version smokes Pytorch in the graphs above(11-13x more throughput). I would like to say Torch is super convenient to use but for raw optimization, much more has to be done. 

Basically the throughput and speedup is signicantly faster in C++ in the Kernel computing

Full Runtime comparison: 
C++: 3307.51 ms
PyTorch 38682.85 ms

11.7x speedup when accounting for cpu + gpu data transfers + all operations


C++ written with only CUDA libary calls for cpu/gpu data transfer.

Future Considerations
- look into deriving cudaMalloc and cudaMemcpy to get full control how neurons react in data transfer  
- take advantage of unifed memory(I only use 1 hardware device so maybe no point)
- potentially use tiling with matrix multiplication to full extent.
- Use different variations of softmax beside from hardcoding output layer. 
   



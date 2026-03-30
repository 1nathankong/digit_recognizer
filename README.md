Learning how to code without Generative AI and learn how to do Mathmatical modeling step by step. This example is an implementation of a Feedforward Neural Network using Backpropagation and Gradient Descent. 


Dataset I used: https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv

Update status:

First got basic Neural Network with backprop working with 95% accuracy in 3000 iterations

Rewrote my Gradient Descent Function as a SGD so I can look at 1 small subset iteration when training. 

Planning C++ rewrite
    /include
        data_loader.hpp - Load the MNIST dataset in C++(done)
        layer.hpp - weights and bias declared and also declaration of compuations done with GPU (in progress)

    /src
        data_loader.cpp - read csv data from data_loader and format to use for Neural Network. 
        loop.cpp - deal with feedforward and back propagation logic
        layer.cu - raw computations done on GPU
        main.cpp - all files get called here and executes. allocate and deallocate pointers and free memory after run ends
        

    CMakeLists.txt - compiler code to setup model in C++





    



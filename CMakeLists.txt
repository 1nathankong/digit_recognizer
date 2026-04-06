cmake_minimum_required(VERSION 3.18)
project(DIGIT_RECOGNIZER LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(include)

add_executable(DIGIT_RECOGNIZER 
    src/main.cu
    src/layer.cu
    src/data_loader_train.cpp
)

target_link_libraries(DIGIT_RECOGNIZER PRIVATE
    cudart
)

set_target_properties(DIGIT_RECOGNIZER PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
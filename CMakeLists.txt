cmake_minimum_required(VERSION 3.10)
project(DIGIT_RECOGNIZER)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(DIGIT_RECOGNIZER src/data_loader_train.cpp)
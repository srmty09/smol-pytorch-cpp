cmake_minimum_required(VERSION 3.10)

# Project name
project(SmolPyTorchCPP)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directory
include_directories(include)
# Include directory
include_directories(helper)

# Create executable
add_executable(smolpytorch
    main.cpp
    src/tensor.cpp
    helper/helper_functions.cpp
)

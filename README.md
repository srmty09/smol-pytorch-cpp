# Smol-PyTorch-CPP

A lightweight C++ deep learning library inspired by PyTorch, designed for educational purposes and small-scale machine learning projects.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Tensor Operations
- **Element-wise operations**: Addition, Subtraction, Multiplication, Division
- **Mathematical functions**: Exponential, Square root, Power, Absolute value, and more
- **Aggregation functions**: Sum, Mean, Maximum, Minimum
- **Scalar operations**: Add, Subtract, Divide, Multiply with scalars
- **Matrix operations**: Matrix multiplication with automatic shape validation

### Autograd
- **Automatic differentiation**: Basic support for gradient calculation through backpropagation

### Activation Functions
- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Sigmoid activation function
- **Tanh**: Hyperbolic tangent
- **Softmax**: Numerically stable softmax implementation

## Installation

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher

### Build Instructions

```bash
git clone https://github.com/srmty09/smol-pytorch-cpp.git
cd smol-pytorch-cpp
mkdir build
cd build
cmake ..
make
./smolpytorch
```

## Quick Start

Here is a simple example demonstrating creating tensors, performing operations, and running backpropagation.

```cpp
#include "tensor.hpp"
#include "autograd.hpp"
#include <iostream>

int main() {
    // Create tensors that require gradients
    tensor a({2.0, 3.0}, {1, 2}, true);
    tensor b({4.0, 5.0}, {1, 2}, true);

    // Perform some operations
    tensor c = a.add(b);
    tensor d = c.mul(a);
    tensor e = d.sum();

    // Perform backpropagation starting from the final tensor 'e'
    // The gradient of 'e' with respect to itself is 1
    std::vector<double> grad(e.size_, 1.0);
    e.grad_ = grad;
    
    backward backprop;
    backprop.backward_(e);
    
    // Print gradients of 'a'
    std::cout << "Gradient of a: ";
    for (double val : a.grad_) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

## API Reference

### Tensor Class (`tensor.hpp`)

#### Constructor
```cpp
tensor(std::vector<double> data, std::vector<ll> shape, bool need_grad = false);
```

#### Element-wise Operations
```cpp
tensor add(tensor& b);
tensor sub(tensor& b);
tensor mul(tensor& b);
tensor div(tensor& b);
```

#### Mathematical and Activation Functions
```cpp
tensor relu();
tensor abs();
tensor pow(ll power);
tensor exp();
tensor sqrt();
tensor sigmoid();
tensor tanh();
tensor softmax();
tensor neg();
```

#### Scalar Operations
```cpp
tensor scalar_sub(double num);
tensor scalar_div(double num);
tensor scalar_add(double num);
tensor scalar_mul(double num);
```

#### Aggregation Functions
```cpp
double sum();
double mean();
double max();
double min();
```

#### Utility Functions
```cpp
tensor copy();
std::vector<double> get_data();
```

### Autograd Class (`autograd.hpp`)

The `backward` class is used to perform backpropagation.

#### Methods
```cpp
void backward_(tensor& t);
```
This method computes the gradient of the loss with respect to the tensors in the computation graph. It traverses the graph backwards from the given tensor `t`.

## Project Structure
```
smol-pytorch-cpp/
├── include/
│   ├── tensor.hpp
│   └── autograd.hpp
├── src/
│   ├── tensor.cpp
│   └── autograd.cpp
├── helper/
│   ├── helper_functions.hpp
│   └── helper_functions.cpp
├── main.cpp
├── CMakeLists.txt
└── README.md
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.

## Acknowledgments

- Inspired by PyTorch's elegant API design
- Built for educational purposes in deep learning
- Special thanks to the C++ community for best practices

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Happy coding!** 
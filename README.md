# Smol-PyTorch-CPP

A lightweight, header-only C++ deep learning library inspired by PyTorch, designed for educational purposes and small-scale machine learning projects.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Tensor Operations
- **Element-wise operations**: Addition, Subtraction, Multiplication, Division
- **Mathematical functions**: Exponential, Square root, Power, Absolute value
- **Aggregation functions**: Sum, Mean, Maximum, Minimum
- **Scalar operations**: Add, Subtract, Divide with scalars

### Activation Functions
- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Sigmoid activation function
- **Tanh**: Hyperbolic tangent
- **Softmax**: Numerically stable softmax implementation

### Advanced Features
- **Gradient tracking**: Automatic differentiation support (planned)
- **Memory management**: Efficient tensor copying and management
- **Error handling**: Robust division by zero and edge case handling
- **Numerical stability**: Overflow protection in mathematical operations

## Installation

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- Make or Ninja build system

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/srmty09/smol-pytorch-cpp
cd smol-pytorch-cpp

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make

# Run the test suite
./smolpytorch
```

## Quick Start

### Basic Tensor Operations

```cpp
#include "tensor.hpp"
#include <iostream>

int main() {
    // Create tensors
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    std::vector<long long> shape = {2, 2};
    
    tensor a(data, shape, true);  // Enable gradient tracking
    tensor b(data, shape, true);
    
    // Element-wise operations
    tensor c = a.add(b);  // Addition
    tensor d = a.mul(b);  // Multiplication
    
    // Mathematical functions
    a.exp();    // Exponential
    b.sqrt();   // Square root
    
    // Activation functions
    tensor h = a.copy();
    h.relu();   // ReLU activation
    h.sigmoid(); // Sigmoid activation
    h.tanh();   // Tanh activation
    
    return 0;
}
```

### Neural Network Building (Planned)

```cpp
// Future implementation
Sequential model({
    new Linear(784, 128),
    new ReLU(),
    new Linear(128, 10),
    new Softmax()
});

tensor output = model.forward(input);
```

## API Reference

### Tensor Class

#### Constructors
```cpp
tensor();                                    // Null tensor
tensor(vector<double> data, vector<ll> shape, bool need_grad = false);
```

#### Element-wise Operations
```cpp
tensor add(tensor b);    // Element-wise addition
tensor sub(tensor b);    // Element-wise subtraction
tensor mul(tensor b);    // Element-wise multiplication
tensor div(tensor b);    // Element-wise division
```

#### Mathematical Functions
```cpp
void exp();              // Exponential
void sqrt();             // Square root
void pow(ll power);      // Power function
void abs();              // Absolute value
```

#### Activation Functions
```cpp
void relu();             // ReLU activation
void sigmoid();          // Sigmoid activation
void tanh();             // Tanh activation
void softmax();          // Softmax (numerically stable)
```

#### Aggregation Functions
```cpp
double sum();            // Sum of all elements
double mean();           // Mean of all elements
double max();            // Maximum value
double min();            // Minimum value
```

#### Utility Functions
```cpp
tensor copy();           // Deep copy
vector<double> get_data(); // Get tensor data
```

## Examples

### Example 1: Basic Tensor Operations
```cpp
#include "tensor.hpp"

int main() {
    // Create tensors
    vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
    vector<double> data2 = {2.0, 3.0, 4.0, 5.0};
    vector<ll> shape = {2, 2};
    
    tensor a(data1, shape, true);
    tensor b(data2, shape, true);
    
    // Perform operations
    tensor c = a.add(b);
    tensor d = a.mul(b);
    
    // Apply activation functions
    c.relu();
    d.sigmoid();
    
    return 0;
}
```

### Example 2: Mathematical Operations
```cpp
#include "tensor.hpp"

int main() {
    vector<double> data = {-2.0, 0.0, 2.0, 4.0};
    vector<ll> shape = {2, 2};
    
    tensor t(data, shape, false);
    
    // Chain mathematical operations
    t.abs();     // Absolute value
    t.pow(2);    // Square
    t.sqrt();    // Square root
    t.exp();     // Exponential
    
    return 0;
}
```

### Example 3: Softmax with Large Values
```cpp
#include "tensor.hpp"

int main() {
    // Test numerically stable softmax
    vector<double> data = {1000.0, 1001.0, 999.0, 1002.0};
    vector<ll> shape = {2, 2};
    
    tensor t(data, shape, false);
    t.softmax();  // Handles large values without overflow
    
    return 0;
}
```

## Project Structure

```
smol-pytorch-cpp/
├── CMakeLists.txt              # CMake configuration
├── main.cpp                    # Main test file
├── include/
│   └── tensor.hpp             # Tensor class header
├── src/
│   └── tensor.cpp             # Tensor class implementation
├── helper/
│   ├── helper_functions.hpp   # Helper functions header
│   └── helper_functions.cpp   # Helper functions implementation
├── build/                     # Build directory (generated)
└── README.md                  # This file
```

## Key Implementation Details

### Numerical Stability
- **Softmax**: Implements numerically stable version using max subtraction
- **Division by zero**: Graceful handling with warning messages
- **Negative sqrt**: Skips negative values with warning

### Memory Management
- **Copy operations**: Deep copy implementation for independent tensors
- **Gradient tracking**: Optional gradient vector initialization
- **Stride calculation**: Efficient memory layout for multi-dimensional tensors

### Error Handling
- **Bounds checking**: Prevents segmentation faults
- **Input validation**: Checks for valid tensor shapes and sizes
- **Warning messages**: Informative output for edge cases

## Planned Features

- [ ] **Automatic Differentiation**: Backpropagation implementation
- [ ] **Neural Network Layers**: Linear, Convolutional, Pooling layers
- [ ] **Optimizers**: SGD, Adam, RMSprop
- [ ] **Loss Functions**: MSE, Cross-entropy, Binary cross-entropy
- [ ] **GPU Support**: CUDA/OpenCL integration
- [ ] **Model Serialization**: Save/load trained models
- [ ] **Advanced Operations**: Matrix multiplication, transpose, reshape

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch's elegant API design
- Built for educational purposes in deep learning
- Special thanks to the C++ community for best practices

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Happy coding!** 
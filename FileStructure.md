# 📁 miniTorch File Structure

A minimal deep learning framework in C++, inspired by PyTorch.

---

## 🗂️ Project Structure

```text
miniTorch/
├── include/                         # Header files for core components
│   ├── tensor.hpp                   # Tensor data structure and utilities
│   ├── autograd.hpp                 # Autograd engine
│   ├── nn/                          # Neural network components
│   │   ├── layer.hpp                # Base layer class
│   │   ├── linear.hpp               # Fully-connected (Linear) layer
│   │   ├── relu.hpp                 # ReLU activation layer
│   │   ├── conv2d.hpp               # Convolution layer (future)
│   │   └── sequential.hpp           # Sequential container for stacking layers
│   ├── nn/functional/               # Functional API (stateless ops)
│   │   ├── activation.hpp           # relu(), sigmoid(), tanh(), etc.
│   │   └── loss.hpp                 # mse_loss(), cross_entropy()
│   ├── optim/                       # Optimizers
│   │   ├── optimizer.hpp            # Base optimizer interface
│   │   ├── sgd.hpp                  # Stochastic Gradient Descent
│   │   └── adam.hpp                 # Adam optimizer (future)
│   ├── utils/                       # Utility functions
│   │   ├── initializer.hpp          # Xavier, He initialization
│   │   └── dataloader.hpp           # Data loading, batching
│   └── config.hpp                   # Global constants/macros
│
├── src/                             # Implementation source files
│   ├── tensor.cpp
│   ├── autograd.cpp
│   ├── nn/
│   │   ├── layer.cpp
│   │   ├── linear.cpp
│   │   ├── relu.cpp
│   │   ├── conv2d.cpp
│   │   └── sequential.cpp
│   ├── nn/functional/
│   │   ├── activation.cpp
│   │   └── loss.cpp
│   ├── optim/
│   │   ├── optimizer.cpp
│   │   ├── sgd.cpp
│   │   └── adam.cpp
│   ├── utils/
│   │   ├── initializer.cpp
│   │   └── dataloader.cpp
│   └── config.cpp
│
├── tests/                           # Unit tests
│   ├── test_tensor.cpp
│   ├── test_autograd.cpp
│   ├── test_layers.cpp
│   ├── test_functional.cpp
│   ├── test_optim.cpp
│   └── test_training.cpp
│
├── examples/                        # Example programs
│   ├── xor_train.cpp                # Basic XOR training
│   ├── mnist_train.cpp              # Optional: train on MNIST dataset
│   └── test_forward_only.cpp        # Forward-only inference
│
├── main.cpp                         # CLI entry point / test runner
├── CMakeLists.txt                   # CMake build configuration
└── README.md                        # Project overview and usage

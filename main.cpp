#include "tensor.hpp"
#include "autograd.hpp"
#include "matrixmultiplication.hpp"
#include<iostream>
#include<cmath>
using namespace std;
using ll = long long;
#include <cstdlib> 
#include <ctime> 

void print_tensor(const tensor& t, const string& name) {
    cout << name << " data: ";
    for(ll i = 0; i < t.size_; i++) {
        cout << t.data_[i] << " ";
    }
    cout << endl;
}

/*
 * Network Architecture Details:
 *
 * A 4-hidden-layer MLP for regression.
 *
 * - Input Layer: 5 features
 * - Hidden Layer 1: 64 neurons
 *   - W1: (5, 64) -> 320 weights
 * - Hidden Layer 2: 32 neurons
 *   - W2: (64, 32) -> 2048 weights
 * - Hidden Layer 3: 16 neurons
 *   - W3: (32, 16) -> 512 weights
 * - Hidden Layer 4: 8 neurons
 *   - W4: (16, 8) -> 128 weights
 * - Output Layer: 1 neuron
 *   - W5: (8, 1) -> 8 weights
 *
 * Total Weights: 320 + 2048 + 512 + 128 + 8 = 3016
 *
 * Note on Biases:
 * The current implementation uses per-sample biases, meaning the bias tensor shape
 * matches the input batch shape. For a batch of 2000, this adds a large number
 * of parameters. A more standard approach uses a single bias value per neuron.
 * - b1: (2000, 64)
 * - b2: (2000, 32)
 * - b3: (2000, 16)
 * - b4: (2000, 8)
 * - b5: (2000, 1)
 */
int main() {
    cout << "=== Network Architecture ===" << endl;
    cout << "  - Input Layer: 5 features" << endl;
    cout << "  - Hidden Layer 1: 64 neurons (W1: 5x64, 320 weights)" << endl;
    cout << "  - Hidden Layer 2: 32 neurons (W2: 64x32, 2048 weights)" << endl;
    cout << "  - Hidden Layer 3: 16 neurons (W3: 32x16, 512 weights)" << endl;
    cout << "  - Hidden Layer 4: 8 neurons (W4: 16x8, 128 weights)" << endl;
    cout << "  - Output Layer: 1 neuron (W5: 8x1, 8 weights)" << endl;
    cout << "Total Weights: 3016" << endl;
    cout << "Total Parameters: 3137" << endl;
    cout << "==========================" << endl << endl;

    
    vector<double> dataX;
    vector<double> dataY;
    
    for(int i = 0; i < 2000; i++) {
        double x1 = 1.0 + i * 0.05;
        double x2 = 2.0 + i * 0.075;
        double x3 = 0.5 + i * 0.025;
        double x4 = 3.0 + i * 0.06;
        double x5 = 1.5 + i * 0.04;
        
        dataX.push_back(x1);
        dataX.push_back(x2);
        dataX.push_back(x3);
        dataX.push_back(x4);
        dataX.push_back(x5);
        
        double target = 2.0 * x1 + 1.5 * x2 + 0.8 * x3 + 1.2 * x4 + 0.6 * x5 + 0.5;
        dataY.push_back(target);
    }
    
    vector<ll> shapeX = {2000, 5};
    tensor X(dataX, shapeX, true);
    
    vector<ll> shapeY = {2000, 1};
    tensor Y(dataY, shapeY, true);
    
    vector<double> testX_data;
    vector<double> testY_data;
    
    for(int i = 0; i < 500; i++) {
        double x1 = 1.2 + i * 0.04;
        double x2 = 2.1 + i * 0.065;
        double x3 = 0.6 + i * 0.03;
        double x4 = 3.1 + i * 0.055;
        double x5 = 1.6 + i * 0.035;
        
        testX_data.push_back(x1);
        testX_data.push_back(x2);
        testX_data.push_back(x3);
        testX_data.push_back(x4);
        testX_data.push_back(x5);
        
        double target = 2.0 * x1 + 1.5 * x2 + 0.8 * x3 + 1.2 * x4 + 0.6 * x5 + 0.5;
        testY_data.push_back(target);
    }
    
    vector<ll> testX_shape = {500, 5};
    tensor testX(testX_data, testX_shape, true);
    
    vector<ll> testY_shape = {500, 1};
    tensor testY(testY_data, testY_shape, true);
    
    cout << "Training data shape: " << X.shape_[0] << "x" << X.shape_[1] << endl;
    cout << "Test data shape: " << testX.shape_[0] << "x" << testX.shape_[1] << endl;
    
    srand(time(0)); // Seed RNG for different weight initialization each run
    
    vector<double> w1_data;
    for(int i = 0; i < 5 * 64; i++) {
        w1_data.push_back((rand() % 100 - 50) / 1000.0);
    }
    vector<ll> w1_shape = {5, 64};
    tensor W1(w1_data, w1_shape, true);
    
    vector<double> w2_data;
    for(int i = 0; i < 64 * 32; i++) {
        w2_data.push_back((rand() % 100 - 50) / 1000.0);
    }
    vector<ll> w2_shape = {64, 32};
    tensor W2(w2_data, w2_shape, true);
    
    vector<double> w3_data;
    for(int i = 0; i < 32 * 16; i++) {
        w3_data.push_back((rand() % 100 - 50) / 1000.0);
    }
    vector<ll> w3_shape = {32, 16};
    tensor W3(w3_data, w3_shape, true);
    
    vector<double> w4_data;
    for(int i = 0; i < 16 * 8; i++) {
        w4_data.push_back((rand() % 100 - 50) / 1000.0);
    }
    vector<ll> w4_shape = {16, 8};
    tensor W4(w4_data, w4_shape, true);
    
    vector<double> w5_data;
    for(int i = 0; i < 8 * 1; i++) {
        w5_data.push_back((rand() % 100 - 50) / 1000.0);
    }
    vector<ll> w5_shape = {8, 1};
    tensor W5(w5_data, w5_shape, true);
    
    vector<double> b1_data;
    for(int i = 0; i < 2000 * 64; i++) {
        b1_data.push_back(0.0);
    }
    vector<ll> b1_shape = {2000, 64};
    tensor b1(b1_data, b1_shape, true);
    
    vector<double> b2_data;
    for(int i = 0; i < 2000 * 32; i++) {
        b2_data.push_back(0.0);
    }
    vector<ll> b2_shape = {2000, 32};
    tensor b2(b2_data, b2_shape, true);
    
    vector<double> b3_data;
    for(int i = 0; i < 2000 * 16; i++) {
        b3_data.push_back(0.0);
    }
    vector<ll> b3_shape = {2000, 16};
    tensor b3(b3_data, b3_shape, true);
    
    vector<double> b4_data;
    for(int i = 0; i < 2000 * 8; i++) {
        b4_data.push_back(0.0);
    }
    vector<ll> b4_shape = {2000, 8};
    tensor b4(b4_data, b4_shape, true);
    
    vector<double> b5_data;
    for(int i = 0; i < 2000 * 1; i++) {
        b5_data.push_back(0.0);
    }
    vector<ll> b5_shape = {2000, 1};
    tensor b5(b5_data, b5_shape, true);
    
    double learning_rate = 0.0001;
    int epochs = 2000;
    
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        // --- Zero Gradients ---
        set_all(W1.grad_, 0.0);
        set_all(W2.grad_, 0.0);
        set_all(W3.grad_, 0.0);
        set_all(W4.grad_, 0.0);
        set_all(W5.grad_, 0.0);
        set_all(b1.grad_, 0.0);
        set_all(b2.grad_, 0.0);
        set_all(b3.grad_, 0.0);
        set_all(b4.grad_, 0.0);
        set_all(b5.grad_, 0.0);
        
        // --- Forward Pass ---
        tensor h1 = mat_mul(X, W1);
        tensor h1_bias = h1.add(b1);
        tensor h1_relu = h1_bias.relu();
        
        tensor h2 = mat_mul(h1_relu, W2);
        tensor h2_bias = h2.add(b2);
        tensor h2_relu = h2_bias.relu();
        
        tensor h3 = mat_mul(h2_relu, W3);
        tensor h3_bias = h3.add(b3);
        tensor h3_relu = h3_bias.relu();
        
        tensor h4 = mat_mul(h3_relu, W4);
        tensor h4_bias = h4.add(b4);
        tensor h4_relu = h4_bias.relu();
        
        tensor output = mat_mul(h4_relu, W5);
        tensor output_bias = output.add(b5);
        
        // --- Loss Calculation (MSE) ---
        tensor loss_diff = output_bias.sub(Y);
        tensor loss = loss_diff.pow(2);
        double total_loss = loss.sum() / 2000.0;
        
        // --- Backward Pass ---
        set_all(loss.grad_, 1.0);
        backward backprop;
        backprop.backward_(loss);

        // --- Update Weights ---
        double max_grad = 1.0;
        for(ll i = 0; i < W1.data_.size(); i++) {
            double grad = W1.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            W1.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < W2.data_.size(); i++) {
            double grad = W2.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            W2.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < W3.data_.size(); i++) {
            double grad = W3.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            W3.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < W4.data_.size(); i++) {
            double grad = W4.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            W4.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < W5.data_.size(); i++) {
            double grad = W5.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            W5.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < b1.data_.size(); i++) {
            double grad = b1.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            b1.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < b2.data_.size(); i++) {
            double grad = b2.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            b2.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < b3.data_.size(); i++) {
            double grad = b3.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            b3.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < b4.data_.size(); i++) {
            double grad = b4.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            b4.data_[i] -= learning_rate * grad;
        }
        for(ll i = 0; i < b5.data_.size(); i++) {
            double grad = b5.grad_[i];
            if(abs(grad) > max_grad) grad = (grad > 0) ? max_grad : -max_grad;
            b5.data_[i] -= learning_rate * grad;
        }
        if(epoch % 500 == 0) {
            // Recompute forward and RMSE after update
            tensor h1 = mat_mul(X, W1);
            tensor h1_bias = h1.add(b1);
            tensor h1_relu = h1_bias.relu();
            tensor h2 = mat_mul(h1_relu, W2);
            tensor h2_bias = h2.add(b2);
            tensor h2_relu = h2_bias.relu();
            tensor h3 = mat_mul(h2_relu, W3);
            tensor h3_bias = h3.add(b3);
            tensor h3_relu = h3_bias.relu();
            tensor h4 = mat_mul(h3_relu, W4);
            tensor h4_bias = h4.add(b4);
            tensor h4_relu = h4_bias.relu();
            tensor output = mat_mul(h4_relu, W5);
            tensor output_bias = output.add(b5);
            tensor loss_diff = output_bias.sub(Y);
            tensor loss = loss_diff.pow(2);
            double total_loss = loss.sum() / 2000.0;
            cout << "epoch " << epoch << ", RMSE: " << sqrt(total_loss) << endl;
        }
    }
    
    tensor final_h1 = mat_mul(X, W1);
    tensor final_h1_bias = final_h1.add(b1);
    tensor final_h1_relu = final_h1_bias.relu();
    
    tensor final_h2 = mat_mul(final_h1_relu, W2);
    tensor final_h2_bias = final_h2.add(b2);
    tensor final_h2_relu = final_h2_bias.relu();
    
    tensor final_h3 = mat_mul(final_h2_relu, W3);
    tensor final_h3_bias = final_h3.add(b3);
    tensor final_h3_relu = final_h3_bias.relu();
    
    tensor final_h4 = mat_mul(final_h3_relu, W4);
    tensor final_h4_bias = final_h4.add(b4);
    tensor final_h4_relu = final_h4_bias.relu();
    
    tensor final_output = mat_mul(final_h4_relu, W5);
    tensor final_output_bias = final_output.add(b5);
    
    tensor final_loss_diff = final_output_bias.sub(Y);
    tensor final_loss = final_loss_diff.pow(2);
    cout << "Final Training RMSE: " << sqrt(final_loss.sum()/2000) << endl;
    
    cout << "\n=== Test Results ===" << endl;
    vector<double> test_b1_data;
    for(int i = 0; i < 500 * 64; i++) {
        test_b1_data.push_back(0.0);
    }
    vector<ll> test_b1_shape = {500, 64};
    tensor test_b1(test_b1_data, test_b1_shape, true);
    
    vector<double> test_b2_data;
    for(int i = 0; i < 500 * 32; i++) {
        test_b2_data.push_back(0.0);
    }
    vector<ll> test_b2_shape = {500, 32};
    tensor test_b2(test_b2_data, test_b2_shape, true);
    
    vector<double> test_b3_data;
    for(int i = 0; i < 500 * 16; i++) {
        test_b3_data.push_back(0.0);
    }
    vector<ll> test_b3_shape = {500, 16};
    tensor test_b3(test_b3_data, test_b3_shape, true);
    
    vector<double> test_b4_data;
    for(int i = 0; i < 500 * 8; i++) {
        test_b4_data.push_back(0.0);
    }
    vector<ll> test_b4_shape = {500, 8};
    tensor test_b4(test_b4_data, test_b4_shape, true);
    
    vector<double> test_b5_data;
    for(int i = 0; i < 500 * 1; i++) {
        test_b5_data.push_back(0.0);
    }
    vector<ll> test_b5_shape = {500, 1};
    tensor test_b5(test_b5_data, test_b5_shape, true);
    
    tensor test_h1 = mat_mul(testX, W1);
    tensor test_h1_bias = test_h1.add(test_b1);
    tensor test_h1_relu = test_h1_bias.relu();
    
    tensor test_h2 = mat_mul(test_h1_relu, W2);
    tensor test_h2_bias = test_h2.add(test_b2);
    tensor test_h2_relu = test_h2_bias.relu();
    
    tensor test_h3 = mat_mul(test_h2_relu, W3);
    tensor test_h3_bias = test_h3.add(test_b3);
    tensor test_h3_relu = test_h3_bias.relu();
    
    tensor test_h4 = mat_mul(test_h3_relu, W4);
    tensor test_h4_bias = test_h4.add(test_b4);
    tensor test_h4_relu = test_h4_bias.relu();
    
    tensor test_output = mat_mul(test_h4_relu, W5);
    tensor test_output_bias = test_output.add(test_b5);
    
    tensor test_loss_diff = test_output_bias.sub(testY);
    tensor test_loss = test_loss_diff.pow(2);
    cout << "test RMSE: " << sqrt(test_loss.sum()/500) << endl;
    
cout << "\n=== 5 Random y_pred and y_actual ===" << endl;
srand(time(0));

for(int i = 0; i < 5; i++) {
    int idx = rand() % 500;
    cout <<" Pred: " << test_output_bias.data_[idx]
         << " Target: " << testY.data_[idx] << endl;
}

return 0;
}




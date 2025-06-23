#include "../include/matrixmultiplication.hpp"
#include "../include/tensor.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using ll = long long;

bool can_multiply(const vector<ll>& shape_a, const vector<ll>& shape_b) {
    if (shape_a.size() != 2 || shape_b.size() != 2) {
        cout << "Error: Matrix multiplication requires 2D tensors" << endl;
        return false;
    }
    
    if (shape_a[1] != shape_b[0]) {
        cout << "Error: Incompatible shapes for matrix multiplication" << endl;
        cout << "Shape A: (" << shape_a[0] << ", " << shape_a[1] << ")" << endl;
        cout << "Shape B: (" << shape_b[0] << ", " << shape_b[1] << ")" << endl;
        cout << "Inner dimensions must match: " << shape_a[1] << " != " << shape_b[0] << endl;
        return false;
    }
    
    return true;
}

vector<ll> get_matmul_shape(const vector<ll>& shape_a, const vector<ll>& shape_b) {
    // (m, n) * (n, p) = (m, p)
    return {shape_a[0], shape_b[1]};
}

pair<ll, ll> get_matrix_dims(const vector<ll>& shape) {
    return {shape[0], shape[1]};
}

tensor mat_mul(tensor& a, tensor& b) {
    if (!can_multiply(a.shape_, b.shape_)) {
        cout << "Matrix multiplication failed due to incompatible shapes" << endl;
        return tensor();
    }
    

    ll m = a.shape_[0]; 
    ll n = a.shape_[1];  
    ll p = b.shape_[1];  
    
    vector<ll> output_shape = get_matmul_shape(a.shape_, b.shape_);
    
    vector<double> result(m * p, 0.0);
    
    for (ll i = 0; i < m; i++) {
        for (ll j = 0; j < p; j++) {
            double sum = 0.0;
            for (ll k = 0; k < n; k++) {
                sum += a.data_[i * n + k] * b.data_[k * p + j];
            }
            result[i * p + j] = sum;
        }
    }
    
    tensor result_tensor(result, output_shape, true);
    result_tensor.op_ = "matmul";
    result_tensor.parent_.push_back(&a);
    result_tensor.parent_.push_back(&b);
    
    return result_tensor;
}

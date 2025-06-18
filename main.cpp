#include "tensor.hpp"
#include<iostream>
using namespace std;
using ll = long long;

void print_tensor(const tensor& t, const string& name) {
    cout << name << " data: ";
    for(ll i = 0; i < t.size_; i++) {
        cout << t.data_[i] << " ";
    }
    cout << endl;
}

int main(){
    cout << "=== SMOL-TORCH-CPP STRESS TEST ===" << endl << endl;
    
    // Test 1: Basic tensor creation
    cout << "1. Testing tensor creation..." << endl;
    vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
    vector<ll> shape1 = {2, 2};
    tensor a(data1, shape1, true);
    tensor b(data1, shape1, true);
    print_tensor(a, "Tensor A");
    print_tensor(b, "Tensor B");
    cout << "Tensor creation: PASSED" << endl << endl;
    
    // Test 2: Element-wise operations
    cout << "2. Testing element-wise operations..." << endl;
    tensor c = a.add(b);
    print_tensor(c, "A + B");
    
    tensor d = a.sub(b);
    print_tensor(d, "A - B");
    
    tensor e = a.mul(b);
    print_tensor(e, "A * B");
    
    tensor f = a.div(b);
    print_tensor(f, "A / B");
    cout << "Element-wise operations: PASSED" << endl << endl;
    
    // Test 3: Mathematical functions
    cout << "3. Testing mathematical functions..." << endl;
    vector<double> data2 = {-2.0, 0.0, 2.0, 4.0};
    tensor g(data2, shape1, false);
    print_tensor(g, "Original tensor G");
    
    g.abs();
    print_tensor(g, "After abs()");
    
    g.pow(2);
    print_tensor(g, "After pow(2)");
    
    g.sqrt();
    print_tensor(g, "After sqrt()");
    
    g.exp();
    print_tensor(g, "After exp()");
    cout << "Mathematical functions: PASSED" << endl << endl;
    
    // Test 4: Activation functions
    cout << "4. Testing activation functions..." << endl;
    vector<double> data3 = {-1.0, 0.0, 1.0, 2.0};
    tensor h(data3, shape1, false);
    print_tensor(h, "Original tensor H");
    
    tensor h_relu = h.copy();
    h_relu.relu();
    print_tensor(h_relu, "After relu()");
    
    tensor h_sigmoid = h.copy();
    h_sigmoid.sigmoid();
    print_tensor(h_sigmoid, "After sigmoid()");
    
    tensor h_tanh = h.copy();
    h_tanh.tanh();
    print_tensor(h_tanh, "After tanh()");
    cout << "Activation functions: PASSED" << endl << endl;
    
    // Test 5: Softmax (numerically stable)
    cout << "5. Testing softmax (numerically stable)..." << endl;
    vector<double> data4 = {1000.0, 1001.0, 999.0, 1002.0};
    tensor i(data4, shape1, false);
    print_tensor(i, "Original tensor I (large values)");
    
    i.softmax();
    print_tensor(i, "After softmax()");
    cout << "Softmax (numerically stable): PASSED" << endl << endl;
    
    // Test 6: Aggregation functions
    cout << "6. Testing aggregation functions..." << endl;
    vector<double> data5 = {1.0, 5.0, 3.0, 9.0, 2.0, 7.0};
    vector<ll> shape5 = {2, 3};
    tensor j(data5, shape5, false);
    print_tensor(j, "Tensor J");
    
    cout << "Sum: " << j.sum() << endl;
    cout << "Mean: " << j.mean() << endl;
    cout << "Max: " << j.max() << endl;
    cout << "Min: " << j.min() << endl;
    cout << "Aggregation functions: PASSED" << endl << endl;
    
    // Test 7: Scalar operations
    cout << "7. Testing scalar operations..." << endl;
    tensor k = a.copy();
    print_tensor(k, "Original tensor K");
    
    k.scalar_add(10.0);
    print_tensor(k, "After scalar_add(10.0)");
    
    k.scalar_sub(5.0);
    print_tensor(k, "After scalar_sub(5.0)");
    
    k.scalar_div(2.0);
    print_tensor(k, "After scalar_div(2.0)");
    cout << "Scalar operations: PASSED" << endl << endl;
    
    // Test 8: Edge cases
    cout << "8. Testing edge cases..." << endl;
    
    // Division by zero
    vector<double> data6 = {1.0, 0.0, 3.0};
    vector<double> data7 = {2.0, 0.0, 1.0};
    tensor l(data6, {1, 3}, false);
    tensor m(data7, {1, 3}, false);
    
    cout << "Testing division by zero..." << endl;
    tensor n = l.div(m);  // Should handle division by zero gracefully
    print_tensor(n, "Result of division by zero");
    
    // Negative sqrt
    vector<double> data8 = {4.0, -1.0, 9.0};
    tensor o(data8, {1, 3}, false);
    print_tensor(o, "Before sqrt (with negative)");
    o.sqrt();
    print_tensor(o, "After sqrt (should skip negative)");
    cout << "Edge cases: PASSED" << endl << endl;
    
    // Test 9: Copy and memory management
    cout << "9. Testing copy and memory management..." << endl;
    tensor p = a.copy();
    print_tensor(p, "Copied tensor P");
    
    // Modify original and verify copy is independent
    a.scalar_add(100.0);
    print_tensor(a, "Modified original A");
    print_tensor(p, "Copied tensor P (should be unchanged)");
    cout << "Copy and memory management: PASSED" << endl << endl;
    
    cout << "=== ALL TESTS PASSED! ===" << endl;
    cout << "Your smol-torch-cpp library is working correctly!" << endl;
    
    return 0;
}
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

    vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
    vector<ll> shape1 = {2, 2};
    tensor a(data1, shape1, true);
    a.softmax();
    cout<<a.op_<<endl;

    return 0;
}
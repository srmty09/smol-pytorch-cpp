#include "tensor.hpp"
#include "autograd.hpp"
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
vector<double> data1 = {7.0, -3.0, 2.5, 8.0};    // a
vector<double> data2 = {-2.0, 5.0, 4.0, -6.0};   // b
vector<double> data3 = {3.0, -7.0, 1.5, 2.0};    // d

vector<ll> shape = {2, 2};
tensor a(data1, shape, true);
tensor b(data2, shape, true);
tensor d(data3, shape, true);

tensor c = a.add(b);
tensor e = c.mul(d);
tensor l = e.sub(a);
tensor o = l.div(b);
tensor output = o.relu();

set_all(output.grad_, 1.0);
backward backprop;
backprop.backward_(output);
backprop.backward_(o);
backprop.backward_(l);
backprop.backward_(e);
backprop.backward_(c);

cout << "grad of output" << endl;
for(ll i=0;i<output.grad_.size();i++) cout << output.grad_[i] << endl;
cout << "grad of o" << endl;
for(ll i=0;i<o.grad_.size();i++) cout << o.grad_[i] << endl;
cout << "grad of l" << endl;
for(ll i=0;i<l.grad_.size();i++) cout << l.grad_[i] << endl;
cout << "grad of e" << endl;
for(ll i=0;i<e.grad_.size();i++) cout << e.grad_[i] << endl;
cout << "grad of d" << endl;
for(ll i=0;i<d.grad_.size();i++) cout << d.grad_[i] << endl;
cout << "grad of c" << endl;
for(ll i=0;i<c.grad_.size();i++) cout << c.grad_[i] << endl;
cout << "grad of a" << endl;
for(ll i=0;i<a.grad_.size();i++) cout << a.grad_[i] << endl;
cout << "grad of b" << endl;
for(ll i=0;i<b.grad_.size();i++) cout << b.grad_[i] << endl;

return 0;
}




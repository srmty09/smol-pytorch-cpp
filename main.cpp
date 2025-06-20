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
    cout<<"Hello world"<<endl;
    vector<double> data1 = {1.0, 2.0, 3.0, 4.0};
    vector<ll> shape1 = {2, 2};
    tensor a(data1, shape1, true);
    vector<double> data2 = {1.0, 2.0, 3.0, 4.0};
    vector<ll> shape2 = {2, 2};
    tensor b(data2, shape2, true);
    tensor d(data1,shape1,true);
    tensor c = a.add(b);
    tensor e = c.mul(d);
    tensor l = e.sub(a);
    tensor o = l.div(b);

    set_all(o.grad_,1.0);
    backward backprop;
    backprop.backward_(o);
    backprop.backward_(l);
    backprop.backward_(e);
    backprop.backward_(c);

    cout<<"grad of o"<<endl;
    for(ll i=0;i<o.grad_.size();i++){
        cout<<o.grad_[i]<<endl;
    }
    cout<<"grad of l"<<endl;
    for(ll i=0;i<l.grad_.size();i++){
        cout<<l.grad_[i]<<endl;
    }
    cout<<"grad of e"<<endl;
    for(ll i=0;i<e.grad_.size();i++){
        cout<<e.grad_[i]<<endl;
    }
     
    cout<<"grad of d"<<endl;
    for(ll i=0;i<d.grad_.size();i++){
        cout<<d.grad_[i]<<endl;
    }   
    cout<<"grad of c"<<endl;
    for(ll i=0;i<c.grad_.size();i++){
        cout<<c.grad_[i]<<endl;
    }
    cout<<"grad of a"<<endl;
    for(ll i=0;i<a.grad_.size();i++){
        cout<<a.grad_[i]<<endl;
    }
    cout<<"grad of b"<<endl;
    for(ll i=0;i<b.grad_.size();i++){
        cout<<b.grad_[i]<<endl;
    }
}




#include <bits/stdc++.h>
using ll = long long;
class tensor{
public:
    // attributes of the tensor-datatype
    std::vector<double> data_;
    std::vector<ll> shape_;
    bool need_grad;
    std::vector<double> grad_;
    ll size_;
    std::vector<ll> stride_;
    std::string op_="";
    std::vector<std::shared_ptr<tensor>> parent_;

    //constructor
    tensor();
    tensor(std::vector<double> d,std::vector<ll> s, bool need_grad);
    
    //methods tensor support
    tensor add(tensor b);
    tensor sub(tensor b);
    tensor mul(tensor b);
    tensor div(tensor b);
    tensor copy();
    tensor relu();
    tensor abs();
    tensor pow(ll power);
    tensor exp();
    tensor sqrt();
    double sum();
    double mean();
    double max();
    double min();
    tensor sigmoid();
    tensor tanh();
    tensor softmax();
    tensor neg();
    tensor scalar_sub(double num);
    tensor scalar_div(double num);
    tensor scalar_add(double num);
    tensor scalar_mul(double num);
    std::vector<double> get_data();
};
#include <bits/stdc++.h>
using ll = long long;
class tensor{
public:
    std::vector<double> data_;
    std::vector<ll> shape_;
    bool need_grad;
    std::vector<double> grad_;
    ll size_;
    std::vector<ll> stride_;
    std::string op_="";

    tensor();
    tensor(std::vector<double> d,std::vector<ll> s, bool need_grad);
    tensor add(tensor b);
    tensor sub(tensor b);
    tensor mul(tensor b);
    tensor div(tensor b);
    tensor copy();
    std::vector<double> get_data();
    void relu();
    void abs();
    void pow(ll power);
    void exp();
    void sqrt();
    double sum();
    double mean();
    double max();
    double min();
    void sigmoid();
    void tanh();
    void softmax();
    void neg();
    void scalar_sub(double num);
    void scalar_div(double num);
    void scalar_add(double num);
};
#include<bits/stdc++.h>
using namespace std;
#include "tensor.hpp"
using ll=long long ;
#include "../helper/helper_functions.hpp"
#include "cmath"

//null tensor
tensor::tensor(){
    cout<<"error!! can't create the tensor..."<<endl;
}


// tensor creation 
tensor::tensor(vector<double> d,vector<ll> s,bool nd=true):data_(d),shape_(s){
    ll total_element=1;
    for(ll i=0;i<s.size();i++) total_element*=shape_[i];
    size_=total_element;
    cout<<"Tensor created successfully"<<endl;
    stride_=calculating_stride(shape_);
    if(nd==true){
        grad_=vector<double>(d.size(),0.0);
        cout<<"Gradient vector initialized successfully"<<endl;
    }
    else{
        cout<<"No gradient vector"<<endl;
        grad_=vector<double>();
    }

}

//element wise addition
tensor tensor::add(tensor& b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(data_[i]+b.data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_ = "+";
    res_t.parent_.push_back(this);
    res_t.parent_.push_back(&b);
    return res_t;
}

//element wise substraction
tensor tensor::sub(tensor& b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(data_[i]-b.data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_="-";
    res_t.parent_.push_back(this);
    res_t.parent_.push_back(&b);
    return res_t;
}


//element wise multiplication
tensor tensor::mul(tensor& b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(data_[i]*b.data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_="*";
    res_t.parent_.push_back(this);
    res_t.parent_.push_back(&b);
    return res_t;
}

//element wise divison
tensor tensor::div(tensor& b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        if(b.data_[i]==0){
            cout<<"Warning: Division by zero at index " << i << ", setting result to INF" << endl;
            res.push_back(1e308); // Use a large number instead of returning null tensor
        } else {
            res.push_back(data_[i]/b.data_[i]);
        }
    }
    tensor res_t(res,shape_);
    res_t.op_="/";
    res_t.parent_.push_back(this);
    res_t.parent_.push_back(&b);
    return res_t;
}

//relu
tensor tensor::relu(){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        if(data_[i]>0){
            res.push_back(data_[i]);
        }
        else{
            res.push_back(0);
        }
    }
    tensor res_t(res,shape_);
    res_t.op_="relu";
    res_t.parent_.push_back(this);  // Store pointer to original tensor
    return res_t;
}

//abs
tensor tensor::abs(){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        if(data_[i]>0){
            res.push_back(data_[i]);
        }
        else{
            res.push_back(-data_[i]);
        }
    }
    tensor res_t(res,shape_);
    res_t.op_="abs";
    res_t.parent_.push_back(this);
    return res_t;
}

//power
tensor tensor::pow(ll power){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(std::pow(data_[i],power));
    }
    tensor res_t(res,shape_);
    res_t.op_="pow";
    res_t.parent_.push_back(this);
    return res_t;
}

//exp
tensor tensor::exp(){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(std::exp(data_[i]));
    }
    tensor res_t(res,shape_);
    res_t.op_="exp";
    res_t.parent_.push_back(this);
    return res_t;
}

//sqrt
tensor tensor::sqrt(){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        if(data_[i]<0){
            cout<<"no sqrt of negative number"<<endl;
            res.push_back(data_[i]);
        } else {
            res.push_back(std::sqrt(data_[i]));
        }
    }
    tensor res_t(res,shape_);
    res_t.op_="sqrt";
    res_t.parent_.push_back(this);
    return res_t;
}

//sum of all elements
double tensor::sum(){
    double ans=0.0;
    for(ll i=0;i<size_;i++){
        ans+=data_[i];
    }
    return ans;
}

//mean
double tensor::mean(){
    double ans=0.0;
    for(ll i=0;i<size_;i++){
        ans+=data_[i];
    }
    return ans/size_;
}

//max
double tensor::max(){
    double ans=0.0;
    for(ll i=0;i<size_;i++){
        if(data_[i]>ans){
            ans=data_[i];
        }
        else{
            continue;
        }
    }
    return ans;
}

//min
double tensor::min(){
    double ans=0.0;
    for(ll i=0;i<size_;i++){
        if(data_[i]<ans){
            ans=data_[i];
        }
        else{
            continue;
        }
    }
    return ans;
}

//negattion
tensor tensor::neg(){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(-data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_="neg";
    res_t.parent_.push_back(this);
    return res_t;
}

//scalar add
tensor tensor::scalar_add(double num){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(num + data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_="scalar_add";
    res_t.parent_.push_back(this);
    return res_t;
}

//scalar substraction
tensor tensor::scalar_sub(double num){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(num - data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_="scalar_sub";
    res_t.parent_.push_back(this);
    return res_t;
}

//scalar div
tensor tensor::scalar_div(double num){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        if(data_[i]!=0){
            res.push_back(num/data_[i]);
        }
        else{
            res.push_back(1e308);
        }
    }
    tensor res_t(res,shape_);
    res_t.op_="scalar_div";
    res_t.parent_.push_back(this);
    return res_t;
}

//scalar mul
tensor tensor::scalar_mul(double num){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(num * data_[i]);
    }
    tensor res_t(res,shape_);
    res_t.op_="scalar_mul";
    res_t.parent_.push_back(this);
    return res_t;
}

//copy 
tensor tensor::copy(){
    vector<double> copy_data;
    for(ll i=0;i<size_;i++){
        copy_data.push_back(data_[i]);
    }
    op_="copy";    
    return tensor(copy_data,shape_,true);
}

//sigmoid
tensor tensor::sigmoid(){
    tensor temp = this->neg();
    temp = temp.exp();
    temp = temp.scalar_add(1);
    temp = temp.scalar_div(1);
    temp.op_="sigmoid";
    return temp;
}

//tanh
tensor tensor::tanh(){
    vector<double> res;
    for(ll i = 0; i < size_; i++){
        res.push_back(std::tanh(data_[i]));
    }
    tensor res_t(res,shape_);
    res_t.op_="tanh";
    res_t.parent_.push_back(this);
    return res_t;
}

//get data
std::vector<double> tensor::get_data(){
    vector<double> datal;
    for(ll i=0;i<size_;i++){
        datal.push_back(data_[i]);
    }
    return datal;
}

// softmax
tensor tensor::softmax(){
    double max_val = data_[0];
    for(ll i = 1; i < size_; i++){
        if(data_[i] > max_val){
            max_val = data_[i];
        }
    }
    
    vector<double> exp_data;
    double sum = 0.0;
    for(ll i = 0; i < size_; i++){
        double exp_val = std::exp(data_[i] - max_val);
        exp_data.push_back(exp_val);
        sum += exp_val;
    }
    
    vector<double> res;
    for(ll i = 0; i < size_; i++){
        res.push_back(exp_data[i] / sum);
    }
    
    tensor res_t(res,shape_);
    res_t.op_="softmax";
    res_t.parent_.push_back(this);
    return res_t;
}

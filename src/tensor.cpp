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
tensor::tensor(vector<double> d,vector<ll> s,bool nd=false):data_(d),shape_(s){
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
tensor tensor:: add(tensor b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(data_[i]+b.data_[i]);
    }
    tensor res_t(res,shape_);
    return res_t;
}

//element wise substraction
tensor tensor::sub(tensor b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(data_[i]-b.data_[i]);
    }
    tensor res_t(res,shape_);
    return res_t;
}


//element wise multiplication
tensor tensor::mul(tensor b){
    vector<double> res;
    for(ll i=0;i<size_;i++){
        res.push_back(data_[i]*b.data_[i]);
    }
    tensor res_t(res,shape_);
    return res_t;
}

//element wise divison
tensor tensor::div(tensor b){
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
    return res_t;
}

//relu
void tensor::relu(){
    for(ll i=0;i<size_;i++){
        if(data_[i]>0){
            continue;
        }
        else{
            data_[i]=0;
        }
    }
}

//abs
void tensor::abs(){
    for(ll i=0;i<size_;i++){
        if(data_[i]>0){
            continue;
        }
        else{
            data_[i]=-(data_[i]);
        }
    }
}

//power
void tensor::pow(ll power){
    for(ll i=0;i<size_;i++){
        data_[i]=std::pow(data_[i],power);
    }
}

//exp
void tensor::exp(){
    for(ll i=0;i<size_;i++){
        data_[i]=std::exp(data_[i]);
    }
}

//sqrt
void tensor::sqrt(){
    for(ll i=0;i<size_;i++){
        if(data_[i]<0){
            cout<<"no sqrt of negative number"<<endl;
            continue;
        }
        data_[i]=std::sqrt(data_[i]);
    }
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
void tensor::neg(){
    for(ll i=0;i<size_;i++){
        data_[i]=-data_[i];
    } 
}
//scalar substraction
void tensor::scalar_add(double num){
    for(ll i=0;i<size_;i++){
        data_[i]=num+data_[i];
    } 
}
//scalar substraction
void tensor::scalar_sub(double num){
    for(ll i=0;i<size_;i++){
        data_[i]=num-data_[i];
    } 
}
//scalar substraction
void tensor::scalar_div(double num){
    for(ll i=0;i<size_;i++){
        data_[i]=num/data_[i];
    } 
}

//copy 
tensor tensor::copy(){
    vector<double> copy_data;
    for(ll i=0;i<size_;i++){
        copy_data.push_back(data_[i]);
    }
    return tensor(copy_data,shape_,true);
}

//sigmoid
void tensor::sigmoid(){
    this->neg();
    this->exp();
    this->scalar_add(1);
    this->scalar_div(1);
}

//tanh
void tensor::tanh(){
    for(ll i = 0; i < size_; i++){
        data_[i] = std::tanh(data_[i]);
    }
}

//get data
vector<double> tensor::get_data(){
    vector<double> datal;
    for(ll i=0;i<size_;i++){
        datal.push_back(data_[i]);
    }
    return datal;
}

// softmax
void tensor::softmax(){
    // Find the maximum value to prevent overflow
    double max_val = data_[0];
    for(ll i = 1; i < size_; i++){
        if(data_[i] > max_val){
            max_val = data_[i];
        }
    }
    
    // Compute exp(x - max_val) to prevent overflow
    vector<double> exp_data;
    double sum = 0.0;
    for(ll i = 0; i < size_; i++){
        double exp_val = std::exp(data_[i] - max_val);
        exp_data.push_back(exp_val);
        sum += exp_val;
    }
    
    // Normalize by sum
    for(ll i = 0; i < size_; i++){
        data_[i] = exp_data[i] / sum;
    }
}

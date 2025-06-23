#include<bits/stdc++.h>
using namespace std;
using ll = long long;

//calculating strides
vector<ll> calculating_stride(vector<ll> shape){
    vector<ll> stride_vec;
    ll element =1;
    stride_vec.push_back(element);
    for(ll i=shape.size()-1;i>0;i--){
        element*=shape[i];
        stride_vec.push_back(element);
    }
    return stride_vec;
}


//set all elements in a double vector to a number 
void set_all(vector<double>& a, double toset){
    for(ll i=0;i<a.size();i++){
        a[i]+=toset;
    }
}


void grad_flow_for_add(vector<double> &a,vector<double> &b){
    for (ll i=0; i<a.size();i++){
        a[i]+=b[i];
    }
}


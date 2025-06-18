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
    cout<<"stride calculation successful"<<endl;
    return stride_vec;
}
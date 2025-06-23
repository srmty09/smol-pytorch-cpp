#include "tensor.hpp"
#include "autograd.hpp"
#include "../helper/helper_functions.hpp"
#include <set>

void backward::backward_(tensor& b){
    static std::set<tensor*> visited;
    static bool is_top_level = true;
    
    if (visited.count(&b)) return;
    visited.insert(&b);

    std::vector<tensor*> parents = b.parent_;
    if(parents.empty()){
        if (is_top_level) {
            visited.clear();
            is_top_level = true;
        }
        return;
    }
    else{
        if(b.op_=="+"){
            for(int i=0;i<2;i++){
                grad_flow_for_add(parents[i]->grad_,b.grad_);
            }
        }
        if(b.op_=="*"){
            for(ll i=0;i<parents[0]->grad_.size();i++){
                parents[0]->grad_[i]+=b.grad_[i]*parents[1]->data_[i];
            }
            for(ll i=0;i<parents[1]->grad_.size();i++){
                parents[1]->grad_[i]+=b.grad_[i]*parents[0]->data_[i];
            }            
        }
        if(b.op_=="-"){
            for(ll i=0;i<parents[0]->grad_.size();i++){
                parents[0]->grad_[i]+=b.grad_[i];
            }
            for(ll i=0;i<parents[1]->grad_.size();i++){
                parents[1]->grad_[i] += -b.grad_[i];
            }            
        }
        if (b.op_ == "/") {
            tensor* x = parents[0];
            tensor* y = parents[1];
            tensor& z = b;

            for (ll i = 0; i < x->grad_.size(); ++i) {
                if (y->data_[i] != 0) {
                    x->grad_[i] += z.grad_[i] / y->data_[i];
                } else {
                    x->grad_[i] += 0;
                }
            }

            for (ll i = 0; i < y->grad_.size(); ++i) {
                if (y->data_[i] != 0) {
                    y->grad_[i] += z.grad_[i] * (-x->data_[i] / (y->data_[i] * y->data_[i]));
                } else {
                    y->grad_[i] += 0;
                }
            }
        }
        if(b.op_=="relu"){
            for(ll i=0;i<parents[0]->grad_.size();i++){
                if(parents[0]->data_[i]>0){
                        parents[0]->grad_[i]+=b.grad_[i];
                }
                else{
                    parents[0]->grad_[i]+=0;
                }
            }
        }
        if(b.op_=="abs"){
            for(ll i=0;i<parents[0]->grad_.size();i++){
                if(parents[0]->data_[i]>0){
                        parents[0]->grad_[i]+=b.grad_[i];
                }
                else{
                    parents[0]->grad_[i]+=-b.grad_[i];
                }
            }
        }
        if(b.op_=="exp"){
            for(ll i=0;i<parents[0]->grad_.size();i++){
                parents[0]->grad_[i]+=b.grad_[i]*std::exp(parents[0]->data_[i]);
            }
        }

        if(b.op_=="sqrt"){
            for(ll i=0;i<parents[0]->grad_.size();i++){
                parents[0]->grad_[i]+=b.grad_[i]*((1/2)*(std::pow(parents[0]->data_[i],(-3/2))));
            }
        }
        if(b.op_=="pow"){
            ll power = 2;
            for(ll i=0;i<parents[0]->grad_.size();i++){
                parents[0]->grad_[i]+=b.grad_[i]*power*std::pow(parents[0]->data_[i], power-1);
            }
        }
        if(b.op_=="matmul"){
            tensor* A = parents[0];
            tensor* B = parents[1];
            ll m = A->shape_[0];
            ll n = A->shape_[1];
            ll p = B->shape_[1];
            for(ll i = 0; i < m; i++) {
                for(ll k = 0; k < n; k++) {
                    double grad_sum = 0.0;
                    for(ll j = 0; j < p; j++) {
                        grad_sum += b.grad_[i * p + j] * B->data_[k * p + j];
                    }
                    A->grad_[i * n + k] += grad_sum;
                }
            }
            for(ll k = 0; k < n; k++) {
                for(ll j = 0; j < p; j++) {
                    double grad_sum = 0.0;
                    for(ll i = 0; i < m; i++) {
                        grad_sum += A->data_[i * n + k] * b.grad_[i * p + j];
                    }
                    B->grad_[k * p + j] += grad_sum;
                }
            }
        }
        is_top_level = false;
        for (auto* parent : parents) {
            backward_(*parent);
        }
        is_top_level = true;
    }
}
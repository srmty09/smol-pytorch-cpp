#include "tensor.hpp"
#include "autograd.hpp"
#include "../helper/helper_functions.hpp"


void backward::backward_(tensor& b){
    //fetch the parents
    std::vector<tensor*> parents;
    parents = b.parent_;
    if(parents.empty()){
        std::cout<<"No parent found return"<<std::endl;
        return;
    }
    else{
        std::cout<<"parent found!"<<std::endl;
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
                x->grad_[i] += z.grad_[i] / y->data_[i];
            }


            for (ll i = 0; i < y->grad_.size(); ++i) {
                y->grad_[i] += z.grad_[i] * (-x->data_[i] / (y->data_[i] * y->data_[i]));
            }
        }
    }
}
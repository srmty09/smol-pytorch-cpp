#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include "tensor.hpp"
#include "../helper/helper_functions.hpp"
#include <iostream>


class backward{
public:
    void backward_(tensor& b);

};

#endif 
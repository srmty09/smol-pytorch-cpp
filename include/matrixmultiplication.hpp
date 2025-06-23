#ifndef MATRIX_MULTIPLICATION_HPP
#define MATRIX_MULTIPLICATION_HPP

#include "tensor.hpp"
#include <vector>
#include <iostream>
using namespace std;
using ll = long long;


tensor mat_mul(tensor& a, tensor& b);

bool can_multiply(const vector<ll>& shape_a, const vector<ll>& shape_b);

vector<ll> get_matmul_shape(const vector<ll>& shape_a, const vector<ll>& shape_b);

pair<ll, ll> get_matrix_dims(const vector<ll>& shape);

#endif // MATRIX_MULTIPLICATION_HPP

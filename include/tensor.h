// File: include/tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <initializer_list>

class Tensor {
public:
    // Construct tensor with given shape, e.g. {2,3,4}
    Tensor(const std::vector<int>& shape);

    // Number of dimensions
    int ndim() const;

    // Total number of elements
    size_t size() const;

    // Set value at multi-index (use {i,j,k})
    void set(const std::initializer_list<int>& idx, float value);
    void set(const std::vector<int>& idx, float value);

    // Get reference (for read/write)
    float get(const std::initializer_list<int>& idx) const;
    float get(const std::vector<int>& idx) const;

    // Simple print (shows shape and flat data). If 2D, prints as matrix.
    void print() const;
    // New Functions We Are Adding
void reshape(const std::vector<int>& new_shape);
Tensor add(const Tensor& other) const;
Tensor multiply(const Tensor& other) const;
Tensor transpose2D() const;

private:
    std::vector<int> shape_;
    std::vector<size_t> strides_;
    std::vector<float> data_;

    void compute_strides();
    size_t index_from_indices(const std::vector<int>& idx) const;
};

#endif // TENSOR_H
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

private:
    std::vector<int> shape_;
    std::vector<size_t> strides_;
    std::vector<float> data_;

    void compute_strides();
    size_t index_from_indices(const std::vector<int>& idx) const;
};

#endif // TENSOR_H


// File: src/tensor.cpp
#include "tensor.h"
#include <stdexcept>
#include <sstream>

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    if (shape_.empty()) throw std::invalid_argument("Shape must have at least one dimension");
    for (int s : shape_) if (s <= 0) throw std::invalid_argument("All shape dimensions must be > 0");
    compute_strides();
    data_.assign(size(), 0.0f);
}

int Tensor::ndim() const { return static_cast<int>(shape_.size()); }

size_t Tensor::size() const {
    size_t n = 1;
    for (int s : shape_) n *= static_cast<size_t>(s);
    return n;
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    // Row-major: last dimension changes fastest
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= static_cast<size_t>(shape_[i]);
    }
}

size_t Tensor::index_from_indices(const std::vector<int>& idx) const {
    if (idx.size() != shape_.size()) {
        std::stringstream ss; ss << "Index has " << idx.size() << " dims but tensor has " << shape_.size();
        throw std::out_of_range(ss.str());
    }
    size_t offset = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        int ind = idx[i];
        if (ind < 0 || ind >= shape_[i]) {
            std::stringstream ss; ss << "Index " << ind << " out of bounds for axis " << i << " (size " << shape_[i] << ")";
            throw std::out_of_range(ss.str());
        }
        offset += static_cast<size_t>(ind) * strides_[i];
    }
    return offset;
}

void Tensor::set(const std::initializer_list<int>& idx, float value) {
    std::vector<int> v(idx);
    set(v, value);
}

void Tensor::set(const std::vector<int>& idx, float value) {
    size_t off = index_from_indices(idx);
    data_[off] = value;
}

float Tensor::get(const std::initializer_list<int>& idx) const {
    std::vector<int> v(idx);
    return get(v);
}

float Tensor::get(const std::vector<int>& idx) const {
    size_t off = index_from_indices(idx);
    return data_[off];
}

void Tensor::print() const {
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    std::cout << "]\n";

    // If 2D, print nicely as rows
    if (shape_.size() == 2) {
        int rows = shape_[0];
        int cols = shape_[1];
        for (int r = 0; r < rows; ++r) {
            std::cout << "[ ";
            for (int c = 0; c < cols; ++c) {
                std::cout << get({r, c}) << (c + 1 < cols ? ", " : " ");
            }
            std::cout << "]\n";
        }
        return;
    }

    // Otherwise, print flat data with indices (simple and daddu-friendly)
    std::cout << "Data (flat): ";
    for (size_t i = 0; i < data_.size(); ++i) {
        std::cout << data_[i] << (i + 1 < data_.size() ? ", " : "\n");
    }
}


// File: main.cpp
#include <iostream>
#include "tensor.h"

int main() {
    std::cout << "Simple Tensor Demo (float) - Daddu friendly\n";
    // Create a small 2x3 tensor
    Tensor t({2, 3});
    std::cout << "Created tensor of shape {2,3}. Filling values...\n";

    // Fill with numbers 1..6
    int val = 1;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            t.set({i, j}, static_cast<float>(val++));
        }
    }

    std::cout << "Tensor contents:\n";
    t.print();

    // Example get
    std::cout << "Element at (1,2) = " << t.get({1,2}) << "\n";

    // Example 3D tensor
    Tensor t3({2,2,2});
    std::cout << "\nCreated 3D tensor of shape {2,2,2}. Setting some values...\n";
    t3.set({0,0,0}, 1.5f);
    t3.set({1,1,1}, 9.25f);
    t3.print();

    std::cout << "Done.\n";
    return 0;
}

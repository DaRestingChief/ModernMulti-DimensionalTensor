
// How to run:     g++ main.cpp src/tensor.cpp -Iinclude -o run
// then            ./run


// File: main.cpp
#include <iostream>
#include "include/tensor.h"

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


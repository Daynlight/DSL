#pragma once
#include "../DSL/dsl.h"
#include <iostream>

void VectorExample() {
    Vector<int> vec;

    int x = 10;
    vec.pushHead(x);  // pushBack equivalent in circular buffer

    x = 20;
    vec.pushHead(x);

    int value = vec.at(1); // access second element
    std::cout << "Second element: " << value << std::endl;
}

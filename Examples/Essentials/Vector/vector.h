#pragma once
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <vector>
#include "dsl.h"

namespace Examples{
void Vector() {
    fmt::print(fg(fmt::color::aquamarine) , "=== Vector Example ===\n");
    
    // Create vector with int type
    Essentials::Vector<int> vec;

    // push values
    vec.pushHead(10);
    vec.pushHead(20);

    // push back
    vec.pushBack(30);
    vec.pushBack(-20);

    // push at
    vec.pushAt(-1, 56);

    fmt::print(fg(fmt::color::aquamarine) , "Print vector Data: \n");
    const int size = vec.getSize();
    std::vector<int> vec_data(vec.getSize());
    while(vec.getSize())
        vec_data.emplace_back(vec.popBack());

    fmt::print(fg(fmt::color::violet) , "{}\n\n", vec_data);
};
};

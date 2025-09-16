#pragma once
#include <fmt/core.h>
#include <fmt/color.h>
#include "dsl.h"

namespace Examples{
void Vector() {
    Essentials::Vector<int> vec;

    int x = 10;
    vec.pushHead(x);  // pushBack equivalent in circular buffer

    x = 20;
    vec.pushHead(x);

    int value = vec.at(1); // access second element
    fmt::print(fg(fmt::color::blue) , "Second element: {:d}", value);
};
};

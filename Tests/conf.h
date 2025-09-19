#pragma once

// #define LoudTests

#include <functional>
void RUN(std::pair<unsigned int, unsigned int>* result, 
std::function<std::pair<unsigned int, unsigned int>()> fun){
  std::pair<unsigned int, unsigned int> res = fun();
  result->first +=  res.first;
  result->second +=  res.second;
};
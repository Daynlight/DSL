#pragma once
#include "Tests/conf.h"

namespace IntegrationTests{
bool AllTests() {
  std::pair<unsigned int, unsigned int> result(0, 0);


  if(result.first == result.second) 
    fmt::print(fg(fmt::color::sky_blue), "=== IntegrationTest Score: {:d}\\{:d} passed ===\n\n", result.first, result.second);
  else 
    fmt::print(fg(fmt::color::dark_orange), "=== IntegrationTest Score: {:d}\\{:d} passed ===\n\n", result.first, result.second);
  return result.first == result.second;
};
};
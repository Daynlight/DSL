#pragma once
#include "Tests/conf.h"
#include "Essentials/Vector/vector.h"

namespace UnitTests {
bool AllTests() {
  std::pair<unsigned int, unsigned int> result(0, 0);

  RUN(&result, [](){ return Vector(); });

  if(result.first == result.second) 
    fmt::print(fg(fmt::color::sky_blue), "=== UnitTest Score: {:d}\\{:d} passed ===\n\n", result.first, result.second);
  else 
    fmt::print(fg(fmt::color::dark_orange), "=== UnitTest Score: {:d}\\{:d} passed ===\n\n", result.first, result.second);
  return result.first == result.second;
};
};
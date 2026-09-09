// Neural Network
// Copyright 2026 Daynlight
// Licensed under the GNU General.
// See LICENSE file for details.



#include <fmt/base.h>
#include <fmt/color.h>

#include "Examples/Learn.h"
#include "Examples/Operations.h"
#include "Examples/App.h"



int main(){
  fmt::println(fg(fmt::color::dark_golden_rod) , "============================================");
  fmt::println(fg(fmt::color::dark_golden_rod) , "=========== ExamplesTemplatedCPU ===========");
  fmt::println(fg(fmt::color::dark_golden_rod) , "============================================");
  fmt::println("");

  ExamplesTemplatedCPU::Operations::BaseOperations();
  ExamplesTemplatedCPU::Operations::Serialization();
  ExamplesTemplatedCPU::Operations::SerializationNetwork();
  ExamplesTemplatedCPU::Operations::Forward();
  ExamplesTemplatedCPU::Operations::Backprop();
  ExamplesTemplatedCPU::Learn::Minus();
  ExamplesTemplatedCPU::Learn::XOR();
  ExamplesTemplatedCPU::Learn::Func();
  
  ExamplesTemplatedCPU::App::Minus();

  return 0;
};

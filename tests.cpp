#include "Tests/Unit/unit_tests.h"
#include "Tests/Integration/integration_tests.h"
#include "Tests/Regression/regression_tests.h"
#include "Tests/Performance/performance_tests.h"
#include "Tests/Security/security_tests.h"
#include "Tests/Stress/stress_tests.h"
#include "Tests/Soak/soak_tests.h"

int main(){
  int passed = 0;
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "======== Tests ========\n");
  fmt::print(fg(fmt::color::dark_golden_rod) , "=======================\n");
  passed += UnitTests::AllTests();
  passed += IntegrationTests::AllTests();
  passed += RegressionTests::AllTests();
  passed += PerformanceTests::AllTests();
  passed += SecurityTests::AllTests();
  passed += StressTests::AllTests();
  passed += SoakTests::AllTests();

  if(passed != 7) {
    printf("Not all Tests Passed");
    return 1;
  }
  return 0;
}
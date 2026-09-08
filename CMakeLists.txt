# Neural Network
# Copyright 2026 Daynlight
# Licensed under the GNU General.
# See LICENSE file for details.



cmake_minimum_required(VERSION 3.15)

######################### Set Project #########################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NEURAL_NETWORK_BUILD_EXAMPLES "Build Examples" ON)
option(NEURAL_NETWORK_BUILD_TESTS "Build Tests" ON)

########################### Add dependencies ########################### 
find_package(CWindow CONFIG REQUIRED)

########################### add subdirectories ########################### 
add_subdirectory(NeuralNetwork)

if(NEURAL_NETWORK_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

########################### tests ###########################
if(NEURAL_NETWORK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(Tests)
endif()

############# [Install] #############
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS NeuralNetwork
  EXPORT NeuralNetworkTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetwork/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/NeuralNetwork"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inl"
)

install(
  EXPORT NeuralNetworkTargets
  FILE NeuralNetworkTargets.cmake
  NAMESPACE NeuralNetwork::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetwork"
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeuralNetworkConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetwork"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetwork"
)
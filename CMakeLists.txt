cmake_minimum_required(VERSION 3.10)

############################## Set Default Path ##############################
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/output)

######################### Set Project Language #########################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

############################## Add libstdc++ as Static Lib ##############################
if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

########################### add submodules ########################### 
add_subdirectory(vendor/fmt)

add_subdirectory(DSL)

########################### Example Project ########################### 
project(Example LANGUAGES CXX)
set(headers
  DSL/
  vendor/
  Examples/
  ${CMAKE_SOURCE_DIR}/
)
set(src 
  example.cpp
)
add_executable(Example ${src} ${headers})
target_include_directories(Example PRIVATE ${headers})
target_link_libraries(Example PRIVATE DSL)
target_link_libraries(Example PRIVATE fmt)

########################### Tests Project ########################### 
project(Tests LANGUAGES CXX)
set(headers
  DSL/
  vendor/
  Tests/
  ${CMAKE_SOURCE_DIR}/
)
set(src 
  tests.cpp
)
add_executable(Tests ${src} ${headers})
target_include_directories(Tests PRIVATE ${headers})
target_link_libraries(Tests PRIVATE DSL)
target_link_libraries(Tests PRIVATE fmt)

enable_testing()
add_test(NAME Tests COMMAND Tests)

############################## Set for VS ##############################
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Main)
set_property(TARGET Example PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}$<CONFIG>") 

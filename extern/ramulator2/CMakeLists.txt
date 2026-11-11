cmake_minimum_required(VERSION 3.20)
project(Ramulator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(ramulator SHARED 
resources/ndp_wrappers/ramulator2.cc
resources/ndp_wrappers/ramulator2.hh)

target_include_directories(ramulator 
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/resources/ndp_wrappers
PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(ramulator PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(
  ramulator 
  PUBLIC yaml-cpp
  PUBLIC spdlog
  PUBLIC ${CONAN_LIBS}
)

add_executable(ramulator-exe)
target_link_libraries(
ramulator-exe
# PRIVATE -Wl,--whole-archive ramulator -Wl,--no-whole-archive
PRIVATE ramulator
PRIVATE argparse
)

set_target_properties(
ramulator-exe
PROPERTIES
OUTPUT_NAME ramulator2
)

add_subdirectory(src)
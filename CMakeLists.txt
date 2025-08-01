cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(SuffixTreeProject CXX)

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (C++11 or later is required for modern features)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable
add_executable(suffix_tree_test source/test_suffix_tree.cpp source/suffix_tree.h)
add_executable(nsw_test source/test_nsw.cpp source/nsw.h)
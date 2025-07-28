cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(SuffixTreeProject CXX)

# Set C++ standard (C++11 or later is required for modern features)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add all source files (header files are automatically detected through includes)
set(SOURCES
    source/test_suffix_tree.cpp
)

# Create executable
add_executable(suffix_tree_test ${SOURCES})

# Optional: Enable warnings for better code quality
if(MSVC)
    # For Visual Studio
    target_compile_options(suffix_tree_test PRIVATE /W4)
else()
    # For GCC, Clang, etc.
    target_compile_options(suffix_tree_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
    
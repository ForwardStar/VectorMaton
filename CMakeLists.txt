cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(SuffixTreeProject CXX)

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set C++ standard (C++17 or later is required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable
add_executable(sa_test source/test_sa.cpp source/sa.h source/sa.cpp)
add_executable(nsw_test source/test_nsw.cpp source/nsw.h source/nsw.cpp)
add_executable(db_test source/test_vector_db.cpp source/nsw.h source/nsw.cpp source/sa.h source/sa.cpp source/vector_db.h source/vector_db.cpp)
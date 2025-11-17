cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(vectormaton CXX)

find_package(OpenSSL REQUIRED)

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

# Set C++ standard (C++17 or later is required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("./third_party/hnswlib")

# Create executable
add_executable(sa_test source/headers.h source/test_sa.cpp source/sa.h source/sa.cpp)
add_executable(nsw_test source/headers.h source/test_nsw.cpp source/nsw.h source/nsw.cpp)
add_executable(hnsw_test source/headers.h source/test_hnsw.cpp)
add_executable(baseline_test source/headers.h source/test_baseline.cpp source/baseline.h source/baseline.cpp)
add_executable(vectormaton_test source/headers.h source/test_vectormaton.cpp source/set_hash.h source/set_hash.cpp source/nsw.h source/nsw.cpp source/sa.h source/sa.cpp source/vectormaton.h source/vectormaton.cpp)
add_executable(main source/headers.h source/main.cpp source/set_hash.h source/set_hash.cpp source/sa.h source/sa.cpp source/nsw.h source/nsw.cpp source/vectormaton.h source/vectormaton.cpp source/exact.h source/exact.cpp source/baseline.h source/baseline.cpp)

target_link_libraries(vectormaton_test OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(main OpenSSL::SSL OpenSSL::Crypto)
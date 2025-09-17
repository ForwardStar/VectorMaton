cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(SuffixTreeProject CXX)

find_package(OpenSSL REQUIRED)

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (C++17 or later is required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable
add_executable(sa_test source/test_sa.cpp source/sa.h source/sa.cpp)
add_executable(nsw_test source/test_nsw.cpp source/nsw.h source/nsw.cpp)
add_executable(db_test source/test_vector_db.cpp source/set_hash.h source/set_hash.cpp source/nsw.h source/nsw.cpp source/sa.h source/sa.cpp source/vector_db.h source/vector_db.cpp)
add_executable(main source/main.cpp source/set_hash.h source/set_hash.cpp source/sa.h source/sa.cpp source/nsw.h source/nsw.cpp source/vector_db.h source/vector_db.cpp source/exact.h source/exact.cpp source/baseline.h source/baseline.cpp)

target_link_libraries(db_test OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(main OpenSSL::SSL OpenSSL::Crypto)
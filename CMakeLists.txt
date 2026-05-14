cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(vectormaton CXX)

find_package(OpenSSL REQUIRED)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (C++17 or later is required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("./third_party/hnswlib")

option(ENABLE_ACORN "Build the ACORN external baseline wrapper." OFF)
set(ACORN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ACORN" CACHE PATH "Path to the ACORN checkout.")
set(ACORN_BUILD_DIR "${ACORN_ROOT}/build" CACHE PATH "Path to the ACORN build directory.")

# Create executable
add_executable(sa_test source/headers.h source/test_sa.cpp source/sa.h source/sa.cpp)
add_executable(queue_test source/mpmc_queue.h source/test_queue.cpp)
add_executable(hnsw_test source/headers.h source/test_hnsw.cpp)
add_executable(vectormaton_test source/headers.h source/test_vectormaton.cpp source/sa.h source/sa.cpp source/vectormaton.h source/vectormaton.cpp)
add_executable(main source/headers.h source/main.cpp source/sa.h source/sa.cpp source/vectormaton.h source/vectormaton.cpp source/exact.h source/exact.cpp source/opt_query.h source/opt_query.cpp source/pre_filtering.h source/pre_filtering.cpp source/post_filtering.h source/post_filtering.cpp)
add_executable(parameter_study source/headers.h source/parameter_study.cpp source/exact.h source/exact.cpp source/post_filtering.h source/post_filtering.cpp)

target_link_libraries(vectormaton_test OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(main OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(parameter_study OpenSSL::SSL OpenSSL::Crypto)

if(ENABLE_ACORN)
    find_library(ACORN_FAISS_LIBRARY
        NAMES faiss
        PATHS "${ACORN_BUILD_DIR}/faiss"
        NO_DEFAULT_PATH
        REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    add_executable(test_acorn test_acorn.cpp)
    target_include_directories(test_acorn PRIVATE "${ACORN_ROOT}")
    target_link_libraries(test_acorn
        "${ACORN_FAISS_LIBRARY}"
        OpenMP::OpenMP_CXX
        ZLIB::ZLIB
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})
    set_target_properties(test_acorn PROPERTIES
        BUILD_RPATH "${ACORN_BUILD_DIR}/faiss")
endif()

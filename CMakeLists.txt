cmake_minimum_required(VERSION 3.18)
project(lbvh VERSION 0.1.0 DESCRIPTION "Vulkan Spatial Structures" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(example)

option(BUILD_TESTS "Build tests." ON)
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

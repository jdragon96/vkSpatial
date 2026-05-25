cmake_minimum_required(VERSION 3.18)
project(vklbvh_tests LANGUAGES CXX)

find_package(GTest REQUIRED)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(vklbvh_tests ${TEST_SOURCES})

target_link_libraries(vklbvh_tests
        PRIVATE
        vkBVH::vkBVH
        GTest::gtest
        GTest::gtest_main)

target_include_directories(vklbvh_tests PRIVATE
        ${CMAKE_SOURCE_DIR})

target_compile_definitions(vklbvh_tests PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

include(GoogleTest)
gtest_discover_tests(vklbvh_tests)

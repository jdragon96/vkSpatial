cmake_minimum_required(VERSION 3.18)
project(vkspatial_tests LANGUAGES CXX)

find_package(GTest REQUIRED)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(vkspatial_tests ${TEST_SOURCES})

target_link_libraries(vkspatial_tests
        PRIVATE
        vkSpatial::vkSpatial
        GTest::gtest
        GTest::gtest_main)

target_include_directories(vkspatial_tests PRIVATE
        ${CMAKE_SOURCE_DIR})

target_compile_definitions(vkspatial_tests PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

get_target_property(_gtest_loc GTest::gtest IMPORTED_LOCATION)
get_filename_component(_gtest_dir "${_gtest_loc}" DIRECTORY)
set_target_properties(vkspatial_tests PROPERTIES
        BUILD_RPATH "${_gtest_dir}"
        INSTALL_RPATH "${_gtest_dir}")

include(GoogleTest)
gtest_discover_tests(vkspatial_tests DISCOVERY_MODE PRE_TEST)

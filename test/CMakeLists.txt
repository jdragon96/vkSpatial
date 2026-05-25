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

set_target_properties(vkspatial_tests PROPERTIES
        BUILD_RPATH "$<TARGET_FILE_DIR:GTest::gtest>;$<TARGET_FILE_DIR:GTest::gtest_main>"
        INSTALL_RPATH "$<TARGET_FILE_DIR:GTest::gtest>;$<TARGET_FILE_DIR:GTest::gtest_main>")

include(GoogleTest)
gtest_discover_tests(vkspatial_tests DISCOVERY_MODE PRE_TEST)

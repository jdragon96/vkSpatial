cmake_minimum_required(VERSION 3.18)
project(vkbvh LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(VULKAN_SDK "$ENV{VULKAN_SDK}")
find_library(SHADERC_LIB shaderc_combined HINTS "${VULKAN_SDK}/lib")

file(GLOB_RECURSE VKBVH_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/vkSpatial/*.cpp")

add_library(vkSpatial STATIC ${VKBVH_SOURCES})
add_library(vkSpatial::vkSpatial ALIAS vkSpatial)

target_link_libraries(vkSpatial
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect)

target_include_directories(vkSpatial
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

set(VKBVH_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader"
        CACHE PATH "vkSpatial GLSL shader directory" FORCE)

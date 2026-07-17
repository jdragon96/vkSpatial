cmake_minimum_required(VERSION 3.18)
project(vkbvh LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(VULKAN_SDK "$ENV{VULKAN_SDK}")
find_library(SHADERC_LIB shaderc_combined HINTS "${VULKAN_SDK}/lib")

set(VKBVH_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader"
        CACHE PATH "vkSpatial GLSL shader directory" FORCE)

file(GLOB_RECURSE VKCOMMON_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/vkCommon/*.cpp")

add_library(vkCommon STATIC ${VKCOMMON_SOURCES})
add_library(vkCommon::vkCommon ALIAS vkCommon)

target_link_libraries(vkCommon
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect vk-bootstrap::vk-bootstrap)

target_include_directories(vkCommon
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
        PRIVATE
            "${VULKAN_SDK}/include")

target_compile_definitions(vkCommon PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

file(GLOB_RECURSE VKBVH_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/vkSpatial/*.cpp")

add_library(vkSpatial STATIC ${VKBVH_SOURCES})
add_library(vkSpatial::vkSpatial ALIAS vkSpatial)

target_link_libraries(vkSpatial
        PUBLIC vkCommon::vkCommon Vulkan::Vulkan)

target_include_directories(vkSpatial
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

target_compile_definitions(vkSpatial PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

file(GLOB_RECURSE VKRENDER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/vkRender/*.cpp")

add_library(vkRender STATIC ${VKRENDER_SOURCES})
add_library(vkRender::vkRender ALIAS vkRender)

target_link_libraries(vkRender
        PUBLIC vkSpatial::vkSpatial vkCommon::vkCommon Vulkan::Vulkan)

target_include_directories(vkRender
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>)

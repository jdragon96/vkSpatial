cmake_minimum_required(VERSION 3.18)
project(vkbvh LANGUAGES CXX)

find_package(Vulkan REQUIRED)

# Vulkan SDK shaderc (GLSL 런타임 컴파일)
set(VULKAN_SDK "$ENV{VULKAN_SDK}")
if(NOT VULKAN_SDK)
    set(VULKAN_SDK "/Users/sjy/VulkanSDK/1.4.328.1/macOS")
endif()
find_library(SHADERC_LIB shaderc_combined HINTS "${VULKAN_SDK}/lib")

# src/vkBVH/*.cpp + src/vkBVH/common/*.cpp 자동 수집
file(GLOB VKBVH_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/vkBVH/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vkBVH/command/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vkBVH/common/*.cpp")

add_library(vkBVH STATIC ${VKBVH_SOURCES})
add_library(vkBVH::vkBVH ALIAS vkBVH)

target_link_libraries(vkBVH
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect)

target_include_directories(vkBVH
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>         # #include "vkBVH/common/..."
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>              # #include "SPIRV-Reflect/..."
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

set(VKBVH_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader"
        CACHE PATH "vkBVH GLSL shader directory" FORCE)

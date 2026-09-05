cmake_minimum_required(VERSION 3.18)
project(vkbvh LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(VULKAN_SDK "$ENV{VULKAN_SDK}")
find_library(SHADERC_LIB shaderc_combined HINTS "${VULKAN_SDK}/lib")

set(VKBVH_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader"
        CACHE PATH "Engine GLSL shader directory" FORCE)

add_subdirectory(BVH)
add_subdirectory(Mesh)
add_subdirectory(Engine)
add_subdirectory(TSDF)
add_subdirectory(GlobalRegistration)
add_subdirectory(LocalRegistration)
add_subdirectory(Realsense)
add_subdirectory(Pipeline)


cmake_minimum_required(VERSION 3.18)
project(vkbvh LANGUAGES CXX)

find_package(Vulkan REQUIRED)

set(VULKAN_SDK "$ENV{VULKAN_SDK}")
find_library(SHADERC_LIB shaderc_combined HINTS "${VULKAN_SDK}/lib")

set(VKBVH_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shader"
        CACHE PATH "Engine GLSL shader directory" FORCE)

add_subdirectory(Engine)
add_subdirectory(TSDF)

# Engine::Spatial keeps the BVH family, the iso-surface extractors, and AdaptiveVoxelGrid; the
# TSDF data structures those consume now live in the TSDF domain. Wired here rather than inside
# Engine/CMakeLists.txt because the TSDF target does not exist until the line above.
target_link_libraries(EngineSpatial PUBLIC TSDF)

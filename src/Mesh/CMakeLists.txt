# Mesh domain library: the Marching-Cubes family of iso-surface extractors (CPU + GPU) and the
# framework they share -- VoxelField in, SurfaceMesh out. A sibling of Engine/, TSDF/, BVH/.
# Kernels live beside the CPP that loads them.
file(GLOB_RECURSE MESH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(Mesh STATIC ${MESH_SOURCES})
add_library(Mesh::Mesh ALIAS Mesh)

# TSDF is PUBLIC: VoxelField reads AdvancedEntry to build a field from a reconstructed map.
target_link_libraries(Mesh
        PUBLIC
        Engine::Core
        Engine::Compute
        TSDF
        Vulkan::Vulkan)

target_include_directories(Mesh
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

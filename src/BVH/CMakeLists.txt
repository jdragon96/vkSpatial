# BVH domain library: the binary LBVH, the wide BVH, and the spatial-index factory over them.
# A sibling of Engine/, TSDF/, and Pipeline/. Kernels live beside the CPP that loads them.
file(GLOB_RECURSE BVH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(BVH STATIC ${BVH_SOURCES})
add_library(BVH::BVH ALIAS BVH)

target_link_libraries(BVH
        PUBLIC
        Engine::Core
        Engine::Compute
        Vulkan::Vulkan)

target_include_directories(BVH
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

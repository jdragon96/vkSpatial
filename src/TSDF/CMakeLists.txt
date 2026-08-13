# TSDF domain library. A sibling of Engine/ (not a submodule of it): src/TSDF/, src/Engine/,
# and future src/Registration/ sit at the same level, and namespaces mirror those paths.
file(GLOB_RECURSE TSDF_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(TSDF STATIC ${TSDF_SOURCES})
add_library(TSDF::TSDF ALIAS TSDF)

target_link_libraries(TSDF
        PUBLIC Engine::Core Engine::Compute Engine::Spatial Vulkan::Vulkan)

target_include_directories(TSDF
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

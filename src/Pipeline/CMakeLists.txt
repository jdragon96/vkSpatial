# Reconstruction pipeline: the Track/Map worker stages (acquisition strategies, ICP alignment
# commands, async TSDF mapping) and the Pipeline coordinator that owns them. A sibling of Engine/
# and TSDF/, matching how the namespaces read.
file(GLOB_RECURSE PIPELINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(EnginePipeline STATIC ${PIPELINE_SOURCES})
add_library(Pipeline ALIAS EnginePipeline)

# Engine::Render is PRIVATE: only Render/RenderThread.cpp uses it, and RenderThread.h forward-
# declares the render types, so consumers (and the headless tests) don't pull in GLFW/render.
target_link_libraries(EnginePipeline
        PUBLIC Engine::Features Engine::Core TSDF Mesh BVH Ceres::ceres
        PRIVATE Engine::Render imgui)

target_include_directories(EnginePipeline
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

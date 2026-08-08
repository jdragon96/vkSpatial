file(GLOB_RECURSE ENGINE_CORE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/Core/*.cpp")

add_library(EngineCore STATIC ${ENGINE_CORE_SOURCES})
add_library(Engine::Core ALIAS EngineCore)

target_link_libraries(EngineCore
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect vk-bootstrap::vk-bootstrap)

target_include_directories(EngineCore
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/vma>
        PRIVATE
            "${VULKAN_SDK}/include")

target_compile_definitions(EngineCore PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

file(GLOB_RECURSE ENGINE_RENDER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/Render/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")

add_library(EngineRender STATIC ${ENGINE_RENDER_SOURCES})
add_library(Engine::Render ALIAS EngineRender)

target_link_libraries(EngineRender
        PUBLIC Engine::Core Vulkan::Vulkan
        PRIVATE glfw)

target_include_directories(EngineRender
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

file(GLOB_RECURSE ENGINE_COMPUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Compute/*.cpp")

add_library(EngineCompute STATIC ${ENGINE_COMPUTE_SOURCES})
add_library(Engine::Compute ALIAS EngineCompute)

target_link_libraries(EngineCompute
        PUBLIC Engine::Core Vulkan::Vulkan)

target_include_directories(EngineCompute
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/vma>
        PRIVATE
            "${VULKAN_SDK}/include")

file(GLOB_RECURSE ENGINE_SPATIAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Spatial/*.cpp")

add_library(EngineSpatial STATIC ${ENGINE_SPATIAL_SOURCES})
add_library(Engine::Spatial ALIAS EngineSpatial)

target_link_libraries(EngineSpatial
        PUBLIC Engine::Core Engine::Compute Vulkan::Vulkan)

target_include_directories(EngineSpatial
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

# SLAM backend (pose graph, loop closure, map correction). Pure Eigen math — no
# Vulkan dependency — so it builds and unit-tests independently of the GPU stack.
file(GLOB_RECURSE ENGINE_BACKEND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Backend/*.cpp")

add_library(EngineBackend STATIC ${ENGINE_BACKEND_SOURCES})
add_library(Engine::Backend ALIAS EngineBackend)

target_include_directories(EngineBackend
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3)

file(GLOB_RECURSE ENGINE_FEATURES_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Features/*.cpp")
add_library(EngineFeatures STATIC ${ENGINE_FEATURES_SOURCES})
add_library(Engine::Features ALIAS EngineFeatures)
target_link_libraries(EngineFeatures PUBLIC Eigen3::Eigen)
target_include_directories(EngineFeatures
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3)

# Reconstruction pipeline: the Track/Map worker stages (acquisition strategies, ICP alignment
# commands, async TSDF mapping) and the Pipeline coordinator that owns them. Self-contained — it
# composes Engine::Spatial (TSDF), Engine::Features (ICP/registration), and Engine::Core (device),
# plus the header-only utilities (Channel/Mailbox/StageProfiler/PointCloudIO) via the src include dir.
file(GLOB_RECURSE ENGINE_PIPELINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Pipeline/*.cpp")

add_library(EnginePipeline STATIC ${ENGINE_PIPELINE_SOURCES})
add_library(Engine::Pipeline ALIAS EnginePipeline)

# Engine::Render is PRIVATE: only the Render/ stage (RenderThread.cpp) uses it, and RenderThread.h
# forward-declares all render types, so consumers (and the headless tests) don't pull in GLFW/render.
target_link_libraries(EnginePipeline
        PUBLIC Engine::Spatial Engine::Features Engine::Core Ceres::ceres
        PRIVATE Engine::Render)

target_include_directories(EnginePipeline
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

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

file(GLOB_RECURSE ENGINE_SPATIAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Spatial/*.cpp")

add_library(EngineSpatial STATIC ${ENGINE_SPATIAL_SOURCES})
add_library(Engine::Spatial ALIAS EngineSpatial)

target_link_libraries(EngineSpatial
        PUBLIC Engine::Core Vulkan::Vulkan)

target_include_directories(EngineSpatial
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")

cmake_minimum_required(VERSION 3.18)
project(vkspatial_examples LANGUAGES CXX)

add_executable(voxel_wang_mc voxel_wang_mc.cpp)

target_link_libraries(voxel_wang_mc PRIVATE vkSpatial::vkSpatial)

# Shader dir is already baked into vkSpatial (PRIVATE define), but examples
# may need it if they call Build() directly in the future.
target_compile_definitions(voxel_wang_mc PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

add_executable(wide_bvh_radius wide_bvh_radius.cpp)
target_link_libraries(wide_bvh_radius PRIVATE vkSpatial::vkSpatial)
target_compile_definitions(wide_bvh_radius PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

add_executable(native_raytrace native_raytrace.cpp)
target_link_libraries(native_raytrace PRIVATE vkSpatial::vkSpatial glfw)

add_executable(bvh_shadow_room bvh_shadow_room.cpp)
target_link_libraries(bvh_shadow_room PRIVATE vkRender::vkRender glfw)

add_executable(bvh_path_tracer
        bvh_path_tracer.cpp
        "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
target_link_libraries(bvh_path_tracer PRIVATE vkRender::vkRender glfw)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    set(CUBE_RENDER_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(CUBE_RENDER_VERT_SPV "${CUBE_RENDER_SHADER_OUTPUT_DIR}/cube.vert.spv")
    set(CUBE_RENDER_FRAG_SPV "${CUBE_RENDER_SHADER_OUTPUT_DIR}/cube.frag.spv")
    set(REALTIME_SHADOW_DEPTH_SPV "${CUBE_RENDER_SHADER_OUTPUT_DIR}/realtime_shadow_depth.vert.spv")
    set(REALTIME_SHADOW_SCENE_VERT_SPV "${CUBE_RENDER_SHADER_OUTPUT_DIR}/realtime_shadow_scene.vert.spv")
    set(REALTIME_SHADOW_SCENE_FRAG_SPV "${CUBE_RENDER_SHADER_OUTPUT_DIR}/realtime_shadow_scene.frag.spv")

    add_custom_command(
            OUTPUT "${CUBE_RENDER_VERT_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/cube.vert"
                    -o "${CUBE_RENDER_VERT_SPV}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cube.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${CUBE_RENDER_FRAG_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/cube.frag"
                    -o "${CUBE_RENDER_FRAG_SPV}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cube.frag"
            VERBATIM)

    add_custom_target(cube_render_shaders
            DEPENDS "${CUBE_RENDER_VERT_SPV}" "${CUBE_RENDER_FRAG_SPV}")

    add_executable(cube_render
            cube_render.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(cube_render cube_render_shaders)
    target_link_libraries(cube_render PRIVATE vkRender::vkRender glfw)
    target_compile_definitions(cube_render PRIVATE
            CUBE_RENDER_SHADER_DIR=\"${CUBE_RENDER_SHADER_OUTPUT_DIR}\")

    add_custom_command(
            OUTPUT "${REALTIME_SHADOW_DEPTH_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/realtime_shadow_depth.vert"
                    -o "${REALTIME_SHADOW_DEPTH_SPV}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/realtime_shadow_depth.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${REALTIME_SHADOW_SCENE_VERT_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/realtime_shadow_scene.vert"
                    -o "${REALTIME_SHADOW_SCENE_VERT_SPV}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/realtime_shadow_scene.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${REALTIME_SHADOW_SCENE_FRAG_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/realtime_shadow_scene.frag"
                    -o "${REALTIME_SHADOW_SCENE_FRAG_SPV}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/realtime_shadow_scene.frag"
            VERBATIM)

    add_custom_target(realtime_shadow_shaders
            DEPENDS
                "${REALTIME_SHADOW_DEPTH_SPV}"
                "${REALTIME_SHADOW_SCENE_VERT_SPV}"
                "${REALTIME_SHADOW_SCENE_FRAG_SPV}")

    add_executable(realtime_shadow
            realtime_shadow.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(realtime_shadow realtime_shadow_shaders)
    target_link_libraries(realtime_shadow PRIVATE vkRender::vkRender glfw)
    target_compile_definitions(realtime_shadow PRIVATE
            REALTIME_SHADOW_SHADER_DIR=\"${CUBE_RENDER_SHADER_OUTPUT_DIR}\")
else ()
    message(WARNING "cube_render/realtime_shadow examples skipped because glslc was not found")
endif ()

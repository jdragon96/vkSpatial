cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

add_executable(voxel_tsdf_mc voxel_tsdf_mc.cpp)
target_link_libraries(voxel_tsdf_mc PRIVATE Engine::Spatial)
target_compile_definitions(voxel_tsdf_mc PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

add_executable(directional_tsdf_demo directional_tsdf_demo.cpp)
target_link_libraries(directional_tsdf_demo PRIVATE Engine::Spatial)
target_compile_definitions(directional_tsdf_demo PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

add_executable(directional_tsdf_eval directional_tsdf_eval.cpp)
target_link_libraries(directional_tsdf_eval PRIVATE Engine::Spatial)
target_compile_definitions(directional_tsdf_eval PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    set(CUBE_RENDER2_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(CUBE_RENDER2_VERT_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.vert.spv")
    set(CUBE_RENDER2_FRAG_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.frag.spv")
    set(SHADOW_MAP_DEPTH_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/realtime_shadow_depth.vert.spv")
    set(SHADOW_MAP_SCENE_VERT_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/realtime_shadow_scene.vert.spv")
    set(SHADOW_MAP_SCENE_FRAG_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/realtime_shadow_scene.frag.spv")

    add_custom_command(
            OUTPUT "${CUBE_RENDER2_VERT_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/cube.vert"
                    -o "${CUBE_RENDER2_VERT_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/cube.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${CUBE_RENDER2_FRAG_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/cube.frag"
                    -o "${CUBE_RENDER2_FRAG_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/cube.frag"
            VERBATIM)

    add_custom_target(cube_render2_shaders
            DEPENDS "${CUBE_RENDER2_VERT_SPV}" "${CUBE_RENDER2_FRAG_SPV}")

    add_custom_command(
            OUTPUT "${SHADOW_MAP_DEPTH_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/realtime_shadow_depth.vert"
                    -o "${SHADOW_MAP_DEPTH_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/realtime_shadow_depth.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${SHADOW_MAP_SCENE_VERT_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/realtime_shadow_scene.vert"
                    -o "${SHADOW_MAP_SCENE_VERT_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/realtime_shadow_scene.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${SHADOW_MAP_SCENE_FRAG_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/realtime_shadow_scene.frag"
                    -o "${SHADOW_MAP_SCENE_FRAG_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/realtime_shadow_scene.frag"
            VERBATIM)

    add_custom_target(shadow_map_shaders
            DEPENDS
                "${SHADOW_MAP_DEPTH_SPV}"
                "${SHADOW_MAP_SCENE_VERT_SPV}"
                "${SHADOW_MAP_SCENE_FRAG_SPV}")

    add_executable(cube_render2
            cube_render.cpp
            CubePass.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(cube_render2 cube_render2_shaders)
    target_link_libraries(cube_render2 PRIVATE Engine::Render Engine::Core)
    target_compile_definitions(cube_render2 PRIVATE
            CUBE_RENDER2_SHADER_DIR=\"${CUBE_RENDER2_SHADER_OUTPUT_DIR}\")

    add_executable(shadow_map
            ShadowMap.cpp)
    add_dependencies(shadow_map shadow_map_shaders)
    target_link_libraries(shadow_map PRIVATE Engine::Render Engine::Core)
    target_compile_definitions(shadow_map PRIVATE
            SHADOW_MAP_SHADER_DIR=\"${CUBE_RENDER2_SHADER_OUTPUT_DIR}\")
else ()
    message(WARNING "cube_render2 example skipped because glslc was not found")
endif ()

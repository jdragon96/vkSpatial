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

add_executable(bvh_benchmark bvh_benchmark.cpp)
target_link_libraries(bvh_benchmark PRIVATE Engine::Spatial)
target_compile_definitions(bvh_benchmark PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    set(CUBE_RENDER2_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(CUBE_RENDER2_VERT_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.vert.spv")
    set(CUBE_RENDER2_FRAG_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.frag.spv")

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

    add_executable(cube_render2
            cube_render.cpp
            CubePass.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(cube_render2 cube_render2_shaders)
    target_link_libraries(cube_render2 PRIVATE Engine::Render Engine::Core)
    target_compile_definitions(cube_render2 PRIVATE
            CUBE_RENDER2_SHADER_DIR=\"${CUBE_RENDER2_SHADER_OUTPUT_DIR}\")
else ()
    message(WARNING "cube_render2 example skipped because glslc was not found")
endif ()

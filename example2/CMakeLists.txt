cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

# Engine::Spatial example: builds `name` from ARGN, links Engine::Spatial, and injects
# the runtime GLSL shader directory (VKBVH_SHADER_DIR) so the executable can compile the
# .comp shaders at runtime via shaderc.
function(add_spatial_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE Engine::Spatial)
    target_compile_definitions(${name} PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
endfunction()

add_spatial_example(voxel_tsdf_mc voxel_tsdf_mc.cpp)
add_spatial_example(directional_tsdf_demo directional_tsdf_demo.cpp)
add_spatial_example(directional_tsdf_eval directional_tsdf_eval.cpp)
add_spatial_example(directional_tsdf_chair_benchmark directional_tsdf_chair_benchmark.cpp)
add_spatial_example(bvh_benchmark bvh_benchmark.cpp)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    # Compiles the given GLSL shaders (relative to this dir) to SPIR-V in a build folder,
    # wires them as a build dependency of `target`, and injects that folder into the
    # target as the `define` macro so the executable loads the compiled shaders at runtime.
    #   add_compiled_shaders(<target> <DEFINE_NAME> <shader> [<shader> ...])
    function(add_compiled_shaders target define)
        set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
        set(spv_outputs "")
        foreach (shader IN LISTS ARGN)
            set(spv "${out_dir}/${shader}.spv")
            add_custom_command(
                    OUTPUT "${spv}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
                    COMMAND ${GLSLC_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" -o "${spv}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
                    VERBATIM)
            list(APPEND spv_outputs "${spv}")
        endforeach ()
        add_custom_target(${target}_shaders DEPENDS ${spv_outputs})
        add_dependencies(${target} ${target}_shaders)
        target_compile_definitions(${target} PRIVATE ${define}=\"${out_dir}\")
    endfunction()

    add_executable(cube_render2
            cube_render.cpp
            CubePass.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    target_link_libraries(cube_render2 PRIVATE Engine::Render Engine::Core)
    add_compiled_shaders(cube_render2 CUBE_RENDER2_SHADER_DIR cube.vert cube.frag)

    add_executable(shadow_map ShadowMap.cpp)
    target_link_libraries(shadow_map PRIVATE Engine::Render Engine::Core)
    add_compiled_shaders(shadow_map SHADOW_MAP_SHADER_DIR
            realtime_shadow_depth.vert realtime_shadow_scene.vert realtime_shadow_scene.frag)
else ()
    message(WARNING "cube_render2/shadow_map examples skipped because glslc was not found")
endif ()

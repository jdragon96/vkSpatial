cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

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
add_spatial_example(scan_dataset_gen scan_dataset_gen.cpp)

# Registers raw scanData/frame_*.ply frames with Engine::Registration before feeding
# DirectionalTSDF (drops the "already pre-registered" assumption the chair benchmark makes).
add_spatial_example(registration_chair_demo registration_chair_demo.cpp)
target_link_libraries(registration_chair_demo PRIVATE Engine::Registration)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    function(add_compiled_shaders target define)
        set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/shaders")
        set(spv_outputs "")
        foreach (shader IN LISTS ARGN)
            get_filename_component(shader_name "${shader}" NAME)
            set(spv "${out_dir}/${shader_name}.spv")
            set(shader_stage_arg "")
            if (shader_name MATCHES "\\.vert(\\.glsl)?$")
                set(shader_stage_arg "-fshader-stage=vertex")
            elseif (shader_name MATCHES "\\.frag(\\.glsl)?$")
                set(shader_stage_arg "-fshader-stage=fragment")
            elseif (shader_name MATCHES "\\.comp(\\.glsl)?$")
                set(shader_stage_arg "-fshader-stage=compute")
            elseif (shader_name MATCHES "\\.geom(\\.glsl)?$")
                set(shader_stage_arg "-fshader-stage=geometry")
            elseif (shader_name MATCHES "\\.tesc(\\.glsl)?$")
                set(shader_stage_arg "-fshader-stage=tesscontrol")
            elseif (shader_name MATCHES "\\.tese(\\.glsl)?$")
                set(shader_stage_arg "-fshader-stage=tesseval")
            endif ()
            add_custom_command(
                    OUTPUT "${spv}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
                    COMMAND ${GLSLC_EXECUTABLE}
                            ${shader_stage_arg}
                            "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" -o "${spv}"
                    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
                    VERBATIM)
            list(APPEND spv_outputs "${spv}")
        endforeach ()
        add_custom_target(${target}_shaders DEPENDS ${spv_outputs})
        add_dependencies(${target} ${target}_shaders)
        target_compile_definitions(${target} PRIVATE ${define}=\"${out_dir}\")
    endfunction()

    add_executable(
            cube_render2
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

    add_executable(blinn_phong BlinnPhong.cpp)
    target_link_libraries(blinn_phong PRIVATE Engine::Render Engine::Core)
    add_compiled_shaders(blinn_phong BLINN_PHONG_SHADER_DIR
            Shaders/BlinnPhong.vert.glsl Shaders/BlinnPhong.frag.glsl)
else ()
    message(WARNING "cube_render2/shadow_map/blinn_phong examples skipped because glslc was not found")
endif ()

cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

function(add_spatial_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE Engine::Spatial)
    target_compile_definitions(${name} PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
endfunction()

add_spatial_example(voxel_tsdf_mc voxel_tsdf_mc.cpp)

# MrHash-foundation validation: online per-voxel TSDF variance (SimpleTSDF::DownloadVoxels)
# proves the paper's premise (edge/corner variance >> flat/curved) and estimates the memory
# a variance-adaptive grid would save. See docs/MRHASH_VS_DIRECTIONAL_TSDF.md.
add_spatial_example(variance_adaptive_demo variance_adaptive_demo.cpp)
target_link_libraries(variance_adaptive_demo PRIVATE Engine::Core)
add_spatial_example(directional_tsdf_demo directional_tsdf_demo.cpp)
add_spatial_example(directional_tsdf_eval directional_tsdf_eval.cpp)
add_spatial_example(directional_tsdf_chair_benchmark directional_tsdf_chair_benchmark.cpp)

# Real-data (chair scan) E2E validation: SimpleTSDF vs DirectionalTSDF vs
# CompactDirectionalTSDF on scanData/frame_*.ply at a common voxelSize/truncation -- goes
# beyond the synthetic cube/cylinder fixtures used by tsdf_benchmark.cpp. See file header.
add_spatial_example(tsdf_realdata_benchmark tsdf_realdata_benchmark.cpp)

# TiledCompactDirectionalTSDF demo: reconstruct the chair scan at a FINE 1mm voxel (scene spans
# ~827 voxels/axis, beyond any single 512^3 CompactDirectionalTSDF window) via ghost-margin
# tiling, plus a common-voxel validation that tiling matches the single window where both fit.
add_spatial_example(tiled_compact_demo tiled_compact_demo.cpp)
target_link_libraries(tiled_compact_demo PRIVATE Engine::Core)

# Headless SimpleTSDF-vs-DirectionalTSDF benchmark: accuracy (vs analytic ground truth) +
# occupied-storage memory + host wall-clock speed, on the same shape_fixtures.h fixtures used
# by tsdf_feature_compare/variance_adaptive_demo. No render/GUI deps.
add_spatial_example(tsdf_benchmark tsdf_benchmark.cpp)
target_link_libraries(tsdf_benchmark PRIVATE Engine::Core)

# AdaptiveVoxelGrid Task 5: headless measurement demo -- FineCount/CoarseCount, total-voxel
# reduction vs all-fine SimpleTSDF::FilledCount(), and mesh accuracy (maxNearestDist) across a
# percentile sweep, for both the cube and cylinder fixtures. No render/GUI deps.
add_spatial_example(adaptive_voxel_grid_demo adaptive_voxel_grid_demo.cpp)
target_link_libraries(adaptive_voxel_grid_demo PRIVATE Engine::Spatial Engine::Core)

add_spatial_example(bvh_benchmark bvh_benchmark.cpp)
add_spatial_example(scan_dataset_gen scan_dataset_gen.cpp)

# Folder → TSDF evaluator (headless, no render/GUI): read every frame_*.ply in a folder,
# integrate into AdvancedTSDF, extract, and score the surface against ground_truth.ply (RMSE).
# Per-frame camera position is estimated from each cloud, so it works on any oriented-point PLY
# folder (object_scan_viewer output or otherwise). AdvancedTSDF compiles its shaders at runtime.
add_spatial_example(tsdf_folder_eval tsdf_folder_eval.cpp)
target_link_libraries(tsdf_folder_eval PRIVATE Engine::Core)

# Registers raw scanData/frame_*.ply frames with Engine::Registration before feeding
# DirectionalTSDF (drops the "already pre-registered" assumption the chair benchmark makes).
add_spatial_example(registration_chair_demo registration_chair_demo.cpp)
add_spatial_example(tsdf_slice_debug tsdf_slice_debug.cpp)
target_link_libraries(registration_chair_demo PRIVATE Engine::Pipeline)

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

    add_spatial_example(tsdf_viewer tsdf_viewer.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(tsdf_viewer PRIVATE Engine::Render imgui)
    add_compiled_shaders(tsdf_viewer VIEWER_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)

    # Interactive Simple-vs-Directional feature-preservation viewer (Task 3). Windowed: A/B
    # toggle between the two extracted clouds (same camera), error/region color modes, and an
    # ImGui panel with the per-region metric table. `--dump` still runs the Task-2 headless
    # correctness oracle and returns before any window is created.
    add_spatial_example(tsdf_feature_compare tsdf_feature_compare.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(tsdf_feature_compare PRIVATE Engine::Render Engine::Core imgui)
    add_compiled_shaders(tsdf_feature_compare FEATURE_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)

    # Live windowed viewer for the real chair scan: integrates scanData/frame_*.ply into a TSDF
    # (Simple / Directional / Compact-Directional, switchable at runtime) and renders input +
    # extracted-surface point clouds with a trackball camera fit to the chair's ~827mm scale.
    add_spatial_example(chair_tsdf_viewer chair_tsdf_viewer.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(chair_tsdf_viewer PRIVATE Engine::Render imgui Engine::Core)
    add_compiled_shaders(chair_tsdf_viewer CHAIR_VIEWER_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)

    # Object scan viewer: load a mesh (PLY/OBJ), centre it at the origin, orbit a spiral trackball
    # camera, capture per-view world-space point clouds (saved as frame_%04d.ply), and WATCH the
    # scan camera orbit the object live (no TSDF here — reconstruction is tsdf_folder_eval).
    # --no-view = headless capture+save.
    add_spatial_example(object_scan_viewer object_scan_viewer.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(object_scan_viewer PRIVATE Engine::Render imgui Engine::Core)
    add_compiled_shaders(object_scan_viewer SCAN_VIEWER_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)

    # Voxel-fill debugger: integrate a folder of frame PLYs into AdvancedTSDF frame by frame and
    # render the filling voxels (colored by tsdf/weight/fill-frame/direction, weight-thresholded)
    # to debug coverage/holes. --dump = headless per-frame stats.
    add_spatial_example(voxel_fill_debugger voxel_fill_debugger.cpp VoxelFillRenderStrategy.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(voxel_fill_debugger PRIVATE Engine::Render imgui Engine::Core Engine::Pipeline)
    add_compiled_shaders(voxel_fill_debugger VOXDBG_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)
else ()
    message(WARNING "cube_render2/shadow_map/blinn_phong examples skipped because glslc was not found")
endif ()

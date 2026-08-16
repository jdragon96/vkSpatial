cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

function(add_spatial_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE Mesh BVH)
    target_compile_definitions(${name} PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
endfunction()

# MrHash-foundation validation: online per-voxel TSDF variance (SimpleTSDF::DownloadVoxels)
# proves the paper's premise (edge/corner variance >> flat/curved) and estimates the memory
# a variance-adaptive grid would save. See docs/MRHASH_VS_DIRECTIONAL_TSDF.md.

# Real-data (chair scan) E2E validation: SimpleTSDF vs DirectionalTSDF vs
# CompactDirectionalTSDF on scanData/frame_*.ply at a common voxelSize/truncation -- goes

# TiledCompactDirectionalTSDF demo: reconstruct the chair scan at a FINE 1mm voxel (scene spans
# ~827 voxels/axis, beyond any single 512^3 CompactDirectionalTSDF window) via ghost-margin
# tiling, plus a common-voxel validation that tiling matches the single window where both fit.

# Headless SimpleTSDF-vs-DirectionalTSDF benchmark: accuracy (vs analytic ground truth) +
# occupied-storage memory + host wall-clock speed, on the same shape_fixtures.h fixtures used

# AdaptiveVoxelGrid Task 5: headless measurement demo -- FineCount/CoarseCount, total-voxel
# reduction vs all-fine SimpleTSDF::FilledCount(), and mesh accuracy (maxNearestDist) across a
# percentile sweep, for both the cube and cylinder fixtures. No render/GUI deps.

add_spatial_example(bvh_benchmark bvh_benchmark.cpp)
add_spatial_example(scan_dataset_gen scan_dataset_gen.cpp)

# Folder → TSDF evaluator (headless, no render/GUI): read every frame_*.ply in a folder,
# integrate into AdvancedTSDF, extract, and score the surface against ground_truth.ply (RMSE).
# Per-frame camera position is estimated from each cloud, so it works on any oriented-point PLY
# folder (object_scan_viewer output or otherwise). AdvancedTSDF compiles its shaders at runtime.
add_spatial_example(tsdf_folder_eval tsdf_folder_eval.cpp)
target_link_libraries(tsdf_folder_eval PRIVATE Engine::Core)

# Per-frame view of what the TSDF pipeline did: splitter division, window count, hash
# load, and the share of the map below a weight threshold (a hole predictor).
add_spatial_example(tsdf_pipeline_debugger tsdf_pipeline_debugger.cpp)
target_link_libraries(tsdf_pipeline_debugger PRIVATE Engine::Core Pipeline)

# Registers raw scanData/frame_*.ply frames with Pipeline before feeding
# DirectionalTSDF (drops the "already pre-registered" assumption the chair benchmark makes).

# Headless ICP-quality diagnostic: runs the reconstruction Pipeline over a frame folder with a chosen
# tracker and reports reconstruction drift vs identity + tracker residual RMSE (no window).
add_spatial_example(icp_quality_diag icp_quality_diag.cpp)
target_link_libraries(icp_quality_diag PRIVATE Pipeline Engine::Core)

# RealSense D435 depth-camera capture/replay tool: --record drives a live device through
# DepthRecorder to a directory; --replay reads a recording back through RecordedDepthProvider +
# BackprojectDepth with no camera or SDK required (no window -- headless, like the diagnostics
# above). --record prints a reason and exits when built without librealsense2 (VKBVH_HAS_REALSENSE
# not defined by src/Pipeline/CMakeLists.txt's optional find_package(realsense2)).
add_spatial_example(depth_capture depth_capture.cpp)
target_link_libraries(depth_capture PRIVATE Pipeline)

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

    # Interactive Simple-vs-Directional feature-preservation viewer (Task 3). Windowed: A/B
    # toggle between the two extracted clouds (same camera), error/region color modes, and an
    # ImGui panel with the per-region metric table. `--dump` still runs the Task-2 headless
    # correctness oracle and returns before any window is created.

    # Live windowed viewer for the real chair scan: integrates scanData/frame_*.ply into a TSDF
    # (Simple / Directional / Compact-Directional, switchable at runtime) and renders input +
    # extracted-surface point clouds with a trackball camera fit to the chair's ~827mm scale.

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
    add_spatial_example(voxel_fill_debugger voxel_fill_debugger.cpp VoxelFillRenderStrategy.cpp PointCloudPass.cpp IsosurfaceMeshPass.cpp EdgeOverlayPass.cpp ImGuiPass.cpp)
    target_link_libraries(voxel_fill_debugger PRIVATE Engine::Render imgui Engine::Core Pipeline)
    add_compiled_shaders(voxel_fill_debugger VOXDBG_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag
        Shaders/IsosurfaceMesh.vert.glsl Shaders/IsosurfaceMesh.frag.glsl
        Shaders/EdgeOverlay.vert.glsl Shaders/EdgeOverlay.frag.glsl)

    # Live depth-camera viewer: stream an IDepthProvider (RealSense device or a recording),
    # run each frame through the pipeline's own BackprojectDepth, and render the resulting
    # oriented point cloud. Shows exactly what the reconstruction consumes, with the depth
    # discontinuity filter editable while it runs. --replay works with no hardware.
    add_spatial_example(depth_live_viewer depth_live_viewer.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(depth_live_viewer PRIVATE Engine::Render imgui Engine::Core Pipeline)
    add_compiled_shaders(depth_live_viewer VOXDBG_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)

    # Isosurface (Marching-Cubes family) debug viewer: extracts a surface with a SELECTABLE
    # extractor (mc/mc33/mtet/emc/dc/dmc/cms, see Mesh) over a selectable
    # analytic shape (sphere/box/torus) or, with --dir, a folder of frame_*.ply scans integrated
    # into an AdvancedTSDF -- and renders the resulting triangle mesh so the differences between
    # algorithms (sharp-feature preservation, cracks, topology, rounding) are visible.
    # EdgeOverlayPass draws a toggleable connectivity overlay on top (non-manifold edges red,
    # boundary edges yellow, from Mesh::AnalyzeConnectivity).
    add_executable(isosurface_viewer isosurface_viewer.cpp IsosurfaceMeshPass.cpp EdgeOverlayPass.cpp ImGuiPass.cpp)
    target_link_libraries(isosurface_viewer PRIVATE Engine::Render Mesh Engine::Core Pipeline imgui)
    add_compiled_shaders(isosurface_viewer ISOSURFACE_VIEWER_SHADER_DIR
            Shaders/IsosurfaceMesh.vert.glsl Shaders/IsosurfaceMesh.frag.glsl
            Shaders/EdgeOverlay.vert.glsl Shaders/EdgeOverlay.frag.glsl)
else ()
    message(WARNING "cube_render2/shadow_map/blinn_phong examples skipped because glslc was not found")
endif ()

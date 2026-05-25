cmake_minimum_required(VERSION 3.18)
project(vkspatial_examples LANGUAGES CXX)

add_executable(voxel_wang_mc voxel_wang_mc.cpp)

target_link_libraries(voxel_wang_mc PRIVATE vkSpatial::vkSpatial)

# Shader dir is already baked into vkSpatial (PRIVATE define), but examples
# may need it if they call Build() directly in the future.
target_compile_definitions(voxel_wang_mc PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

add_executable(voxel_tsdf_mc voxel_tsdf_mc.cpp)
target_link_libraries(voxel_tsdf_mc PRIVATE vkSpatial::vkSpatial)
target_compile_definitions(voxel_tsdf_mc PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")

# The vocabulary types more than one library speaks. Depends on Eigen and nothing else in this
# repo, which is what makes it safe for every other target to link -- the moment Common depends on
# a sibling, the sibling can no longer depend on Common.
#
# STATIC rather than header-only because PointCloud::Save/Load are defined out of line: they need
# utilities/PlyFormat.h, and PointCloud.h is included by ~20 translation units that would otherwise
# all pay for <fstream> and <unordered_map>.
file(GLOB_RECURSE COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(EngineCommon STATIC ${COMMON_SOURCES})
add_library(Common ALIAS EngineCommon)

target_link_libraries(EngineCommon PUBLIC Eigen3::Eigen)

target_include_directories(EngineCommon PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        /opt/homebrew/opt/eigen/include/eigen3)

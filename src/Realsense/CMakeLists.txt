# D400-series front end: per-pixel depth confidence and the normal estimation built on it. A
# sibling of Engine/ and TSDF/, not a subfolder of either.
#
# INTERFACE, not STATIC: everything here is header-only so far, and an empty STATIC library is a
# link error on some generators. Give it real .cpp files and this becomes an ordinary add_library.
file(GLOB_RECURSE REALSENSE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if (REALSENSE_SOURCES)
    add_library(EngineRealsense STATIC ${REALSENSE_SOURCES})
    set(REALSENSE_SCOPE PUBLIC)
else ()
    add_library(EngineRealsense INTERFACE)
    set(REALSENSE_SCOPE INTERFACE)
endif ()
add_library(Realsense ALIAS EngineRealsense)

target_link_libraries(EngineRealsense ${REALSENSE_SCOPE} Engine::Core Engine::Compute)

target_include_directories(EngineRealsense ${REALSENSE_SCOPE}
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        /opt/homebrew/opt/eigen/include/eigen3)

# Optional, exactly as src/Pipeline does it: RealSenseD435 compiles to a set of throwing stubs
# without the SDK, so the build works unchanged on a machine with no camera and no librealsense2.
find_package(realsense2 QUIET)
if (realsense2_FOUND)
    target_link_libraries(EngineRealsense ${REALSENSE_SCOPE} realsense2::realsense2)
    target_compile_definitions(EngineRealsense ${REALSENSE_SCOPE} VKBVH_HAS_REALSENSE)
    message(STATUS "librealsense2 found -- Realsense::RealSenseD435 enabled")
else ()
    message(STATUS "librealsense2 not found -- Realsense::RealSenseD435 builds as stubs")
endif ()

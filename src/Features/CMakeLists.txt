# Point-cloud vocabulary and the descriptors built on it.
#
# A sibling of Engine/, not a layer under it: it depends on nothing but Eigen, and it is what the
# two registration modules and Pipeline share. It used to live under Engine/ and that made the
# bottom layer the owner of registration types, which is backwards.
#
# RegistrationTypes.h lives here for the same reason. PointCloud and RegistrationResult are used by
# GlobalRegistration, LocalRegistration, Pipeline and TSDF alike, so neither registration module can
# own them -- this is the lowest place all of them already reach.
file(GLOB_RECURSE FEATURES_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(EngineFeatures STATIC ${FEATURES_SOURCES})
add_library(Features ALIAS EngineFeatures)

target_link_libraries(EngineFeatures PUBLIC Eigen3::Eigen)

target_include_directories(EngineFeatures PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        /opt/homebrew/opt/eigen/include/eigen3)

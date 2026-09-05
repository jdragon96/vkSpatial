# Global registration: FPFH + RANSAC + refinement, composed into one Estimate().
#
# A sibling of Engine/ and Pipeline/, not a subfolder of either. Pipeline manages the threads that
# compose algorithm pipelines; the algorithms themselves live in their own targets.
file(GLOB_RECURSE GLOBAL_REGISTRATION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(EngineGlobalRegistration STATIC ${GLOBAL_REGISTRATION_SOURCES})
add_library(GlobalRegistration ALIAS EngineGlobalRegistration)

# Features carries FPFH and the matching, and Engine::Registration::RegistrationTypes with
# it. Ceres does the non-linear refinement.
target_link_libraries(EngineGlobalRegistration PUBLIC Features Engine::Core Ceres::ceres)

target_include_directories(EngineGlobalRegistration PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        /opt/homebrew/opt/eigen/include/eigen3)

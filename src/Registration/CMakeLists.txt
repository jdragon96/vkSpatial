# Registration: everything that turns point clouds into transforms.
#
#   Frontend/   per-frame registration -- global (FPFH -> RANSAC -> Ceres) and local (point-to-plane
#               ICP, CPU and GPU). The two are ALTERNATIVES, not stages: which one runs is chosen
#               by Pipeline's TrackerRegistry, so there is no class here composing them.
#   Backend/    pose graph, loop closure, Lie -- the optimisation over many frames at once.
#   Features/   FPFH and the matching the frontend's global half is built on.
#
# One target over all of it, because they share a vocabulary rather than a call graph:
# RegistrationConfig/Param/Result, RigidTransform and LocalGrid sit at this folder's root precisely
# because more than one of the three reaches for them.
#
# A sibling of Engine/ and Pipeline/, not a subfolder of either: Pipeline manages the threads that
# compose algorithms, and the algorithms themselves live in their own targets.
file(GLOB_RECURSE REGISTRATION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(EngineRegistration STATIC ${REGISTRATION_SOURCES})
add_library(Registration ALIAS EngineRegistration)

# Common carries PointCloud; Engine::Core and Compute run the GPU solver's kernel; Ceres does the
# global half's non-linear refinement.
target_link_libraries(EngineRegistration PUBLIC Common Engine::Core Engine::Compute Ceres::ceres)

target_include_directories(EngineRegistration PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        /opt/homebrew/opt/eigen/include/eigen3)

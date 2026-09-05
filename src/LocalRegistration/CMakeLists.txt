# Local registration: point-to-plane ICP, CPU and GPU.
#
# The two solvers are ALTERNATIVES, not stages, so there is no pipeline class here to compose them
# -- which of the two runs is chosen by Pipeline's TrackerRegistry, and a wrapper that only forwarded
# that choice would do nothing.
file(GLOB_RECURSE LOCAL_REGISTRATION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(EngineLocalRegistration STATIC ${LOCAL_REGISTRATION_SOURCES})
add_library(LocalRegistration ALIAS EngineLocalRegistration)

# Engine::Core and Compute for the GPU solver's kernel; Features carries the shared
# Engine::Registration types both solvers speak.
target_link_libraries(EngineLocalRegistration PUBLIC Features Engine::Core Engine::Compute)

target_include_directories(EngineLocalRegistration PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        /opt/homebrew/opt/eigen/include/eigen3)

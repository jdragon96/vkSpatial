cmake_minimum_required(VERSION 3.18)
project(vkspatial_example2 LANGUAGES CXX)

find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
if (GLSLC_EXECUTABLE)
    set(CUBE_RENDER2_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    set(CUBE_RENDER2_VERT_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.vert.spv")
    set(CUBE_RENDER2_FRAG_SPV "${CUBE_RENDER2_SHADER_OUTPUT_DIR}/cube.frag.spv")

    add_custom_command(
            OUTPUT "${CUBE_RENDER2_VERT_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/cube.vert"
                    -o "${CUBE_RENDER2_VERT_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/cube.vert"
            VERBATIM)

    add_custom_command(
            OUTPUT "${CUBE_RENDER2_FRAG_SPV}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBE_RENDER2_SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLC_EXECUTABLE}
                    "${CMAKE_SOURCE_DIR}/example/cube.frag"
                    -o "${CUBE_RENDER2_FRAG_SPV}"
            DEPENDS "${CMAKE_SOURCE_DIR}/example/cube.frag"
            VERBATIM)

    add_custom_target(cube_render2_shaders
            DEPENDS "${CUBE_RENDER2_VERT_SPV}" "${CUBE_RENDER2_FRAG_SPV}")

    add_executable(cube_render2
            cube_render.cpp
            "${CMAKE_SOURCE_DIR}/src/utilities/SimpleResource.cpp"
            "${CMAKE_SOURCE_DIR}/src/utilities/Math.cpp")
    add_dependencies(cube_render2 cube_render2_shaders)
    target_link_libraries(cube_render2 PRIVATE Engine::Render Engine::Core glfw)
    target_include_directories(cube_render2 PRIVATE /opt/homebrew/opt/eigen/include/eigen3)
    target_compile_definitions(cube_render2 PRIVATE
            CUBE_RENDER2_SHADER_DIR=\"${CUBE_RENDER2_SHADER_OUTPUT_DIR}\")
else ()
    message(WARNING "cube_render2 example skipped because glslc was not found")
endif ()

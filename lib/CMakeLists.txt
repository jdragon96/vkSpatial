### SPIRV-Reflect ###
add_library(spirv-reflect SHARED ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Reflect/spirv_reflect.h ${CMAKE_CURRENT_SOURCE_DIR}/SPIRV-Reflect/spirv_reflect.cpp)
set_target_properties(spirv-reflect PROPERTIES LINKER_LANGUAGE CXX)
#target_include_directories(spirv-reflect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS spirv-reflect LIBRARY DESTINATION lib)

### Dear ImGui (core + GLFW/Vulkan backends) ###
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)
install(TARGETS imgui LIBRARY DESTINATION lib)

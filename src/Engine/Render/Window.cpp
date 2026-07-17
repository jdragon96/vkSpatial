#include "Engine/Render/Window.h"
#include "Engine/Render/GlfwWindow.h"

#include <stdexcept>

namespace Engine::Render {

    Window::UniquePtr Window::Create(WindowBackend backend, const WindowDescriptor &descriptor) {
        switch (backend) {
            case WindowBackend::GLFW:
                return std::make_unique<GlfwWindow>(descriptor);
        }
        throw std::runtime_error("Window::Create received an unknown WindowBackend");
    }

} // namespace Engine::Render

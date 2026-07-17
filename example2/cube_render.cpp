#include "CubePass.h"

#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/Scene.h"

#include "utilities/Math.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main() {
    try {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {900, 700, "Engine::Render Cube"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);

        vkMath::Quat orientation = vkMath::Quat::Identity();
        vkMath::Vec3 target(0.0f, 0.0f, 0.0f);
        float distance = 4.5f;
        bool dragging = false;
        vkMath::Vec3 lastBall = vkMath::Vec3::Zero();

        auto updateCamera = [&]() {
            const vkMath::Vec3 eye = target + orientation * vkMath::Vec3(0.0f, 0.0f, distance);
            camera.LookAt(eye, target);
        };
        updateCamera();

        const std::string shaderDir = CUBE_RENDER2_SHADER_DIR;
        Engine::Render::RenderGraph graph;
        graph.AddPass(std::make_unique<CubePass>(app.GetContext(), app.GetSwapChain().Format(), shaderDir));

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup trackball(app.GetWindow().Mouse());
        trackball.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button != Engine::Render::MouseButton::Left)
                return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0)
                return;

            vkMath::Vec3 current = vkMath::MapToArcball(
                    e.x, e.y, static_cast<int>(size.width), static_cast<int>(size.height));
            if (!dragging) {
                dragging = true;
                lastBall = current;
                return;
            }

            vkMath::Vec3 axis = lastBall.cross(current);
            if (axis.dot(axis) > 1e-8f) {
                const float w = std::clamp(lastBall.dot(current), -1.0f, 1.0f);
                vkMath::Quat delta = vkMath::Quat(w, axis.x(), axis.y(), axis.z()).normalized();
                orientation = (delta * orientation).normalized();
                updateCamera();
            }
            lastBall = current;
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left)
                dragging = false;
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            distance = std::clamp(distance * std::exp(static_cast<float>(-e.scrollY) * 0.08f), 2.0f, 10.0f);
            updateCamera();
            e.handled = true;
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape)
                app.GetWindow().RequestClose();
        });

        app.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}

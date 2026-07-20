#include "CubePass.h"

#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/Scene.h"

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
        camera.SetOrbit({0.0f, 0.0f, 0.0f}, 4.5f);

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

            if (!camera.IsTrackballDragging()) {
                camera.BeginTrackballDrag(
                        e.x, e.y,
                        static_cast<int>(size.width), static_cast<int>(size.height));
                return;
            }

            camera.DragTrackball(
                    e.x, e.y,
                    static_cast<int>(size.width), static_cast<int>(size.height));
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left)
                camera.EndTrackballDrag();
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            camera.SetDistance(
                    std::clamp(camera.GetDistance() * std::exp(static_cast<float>(-e.scrollY) * 0.08f),
                               2.0f, 10.0f));
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

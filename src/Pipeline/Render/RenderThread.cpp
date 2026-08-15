#include "Pipeline/Render/RenderThread.h"

#include <imgui.h>

#include "Pipeline/Pipeline.h"
#include "Pipeline/Render/RenderStrategy.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"

#include "utilities/Math.h" // vkMath::Vec3

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace Pipeline {

    RenderThread::RenderThread(Config cfg) : m_cfg(std::move(cfg)) {}

    void RenderThread::Run(Pipeline &pipe, IRenderStrategy &strategy) {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {
                static_cast<uint32_t>(m_cfg.width),
                static_cast<uint32_t>(m_cfg.height),
                m_cfg.title};
        Engine::Render::Application app(descriptor);
        Engine::Core::Context &ctx = app.GetContext();

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        Engine::Render::RenderGraph graph;

        strategy.Build(pipe, app, graph);

        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        float extent = 1.0f;
        strategy.CameraFit(center, extent);
        const VkExtent2D ext = app.GetSwapChain().Extent();
        camera.SetPerspective(60.0f * 3.14159265f / 180.0f,
                              float(ext.width) / float(ext.height),
                              std::max(1e-3f, extent * 1e-3f),
                              std::max(10.0f, extent * 6.0f));
        camera.SetOrbit(vkMath::Vec3(center.x(), center.y(), center.z()), extent * 2.0f);

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        // Generic navigation: left-drag orbits (trackball), right-drag pans (translate), scroll zooms,
        // ESC closes. Right-drag panning needs the previous cursor position to form a per-frame delta
        // (the trackball tracks its own state internally; Camera::Pan takes a pixel delta).
        Engine::Render::MouseListenerGroup mouse(app.GetWindow().Mouse());
        bool panning = false;

        // A drag or scroll that starts over an ImGui window belongs to that window. Without this the
        // camera zooms while the operator scrolls a panel, and orbits while they drag a slider.
        // ImGui reports this through WantCaptureMouse; a strategy with no UI leaves it false.
        auto uiHasMouse = [] { return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse; };
        double lastPanX = 0.0, lastPanY = 0.0;
        mouse.Add(
                Engine::Render::MouseEventType::Drag,
                [&](Engine::Render::MouseEvent &e) {
                    // Only gate the START of a drag: once the camera owns it, the cursor may leave
                    // the viewport and the drag must keep working.
                    if (uiHasMouse() && !camera.IsTrackballDragging() && !panning) return;
                    const VkExtent2D s = app.GetWindow().FramebufferSize();
                    if (s.width == 0 || s.height == 0) return;
                    if (e.button == Engine::Render::MouseButton::Left) {
                        if (!camera.IsTrackballDragging()) {
                            camera.BeginTrackballDrag(e.x, e.y, int(s.width), int(s.height));
                            return;
                        }
                        camera.DragTrackball(e.x, e.y, int(s.width), int(s.height));
                        e.handled = true;
                    } else if (e.button == Engine::Render::MouseButton::Right) {
                        if (!panning) { // first event of a drag: anchor the cursor, no move yet
                            panning = true;
                            lastPanX = e.x;
                            lastPanY = e.y;
                            return;
                        }
                        camera.Pan(e.x - lastPanX, e.y - lastPanY, int(s.width), int(s.height));
                        lastPanX = e.x;
                        lastPanY = e.y;
                        e.handled = true;
                    }
                });
        mouse.Add(
                Engine::Render::MouseEventType::ButtonUp,
                [&](Engine::Render::MouseEvent &e) {
                    if (e.button == Engine::Render::MouseButton::Left) camera.EndTrackballDrag();
                    else if (e.button == Engine::Render::MouseButton::Right) panning = false;
                });
        mouse.Add(
                Engine::Render::MouseEventType::Scroll,
                [&](Engine::Render::MouseEvent &e) {
                    if (uiHasMouse()) return;
                    camera.SetDistance(std::clamp(camera.GetDistance() * std::exp(float(-e.scrollY) * 0.08f),
                                                  extent * 0.2f, extent * 20.0f));
                    e.handled = true;
                });
        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape) app.GetWindow().RequestClose();
        });

        std::shared_ptr<const ModelSnapshot> shown;
        try {
            while (!app.GetWindow().ShouldClose()) {
                app.GetWindow().PollEvents();
                const VkExtent2D s = app.GetWindow().FramebufferSize();
                if (s.width == 0 || s.height == 0) continue;

                pipe.CheckErrors();

                std::shared_ptr<const ModelSnapshot> latest = pipe.LatestModel();
                if (latest && latest.get() != shown.get()) {
                    shown = latest;
                    strategy.OnModel(shown);
                }
                strategy.OnFrame();

                if (!app.GetRenderer().BeginFrame(s.width, s.height)) continue;
                app.GetRenderer().Render(app.GetView());
                app.GetRenderer().EndFrame();
            }
        } catch (...) {
            vkDeviceWaitIdle(ctx.device);
            pipe.Stop();
            throw;
        }
        vkDeviceWaitIdle(ctx.device);
        pipe.Stop();
    }

} // namespace Pipeline

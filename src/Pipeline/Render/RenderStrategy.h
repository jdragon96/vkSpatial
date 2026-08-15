#pragma once

#include "Pipeline/Types.h" // ModelSnapshot, Eigen

#include <memory>

namespace Engine::Render {
    class Application;
    class RenderGraph;
} // namespace Engine::Render

namespace Pipeline {

    class Pipeline;

    class IRenderStrategy {
    public:
        virtual ~IRenderStrategy() = default;

        // Add render passes to `graph` and wire any UI. Called once, after the window/device exist.
        virtual void Build(Pipeline &pipe, Engine::Render::Application &app,
                           Engine::Render::RenderGraph &graph) = 0;

        // Initial camera framing: scene center + extent, so RenderThread sets up the trackball.
        virtual void CameraFit(Eigen::Vector3f &center, float &extent) const = 0;

        // A new immutable model snapshot arrived — store it and refresh GPU buffers. Called on the
        // render (main) thread; the shared_ptr keeps the snapshot alive for as long as the strategy
        // holds it.
        virtual void OnModel(std::shared_ptr<const ModelSnapshot> snap) = 0;

        // Per-frame hook before rendering (e.g. rebuild buffers after a UI change). Default no-op.
        virtual void OnFrame() {}
    };

} // namespace Pipeline

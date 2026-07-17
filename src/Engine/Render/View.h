#pragma once

#include "Engine/Render/RenderTypes.h"

#include <memory>

namespace Engine::Render {

    class Camera;
    class RenderGraph;
    class Scene;

    class View {
    public:
        using UniquePtr = std::unique_ptr<View>;

        void SetScene(Scene *scene) { m_scene = scene; }
        void SetCamera(Camera *camera) { m_camera = camera; }
        void SetRenderGraph(RenderGraph *renderGraph) { m_renderGraph = renderGraph; }
        void SetViewport(Viewport viewport) { m_viewport = viewport; }
        void SetClearOptions(ClearOptions clearOptions) { m_clearOptions = clearOptions; }

        Scene *GetScene() const { return m_scene; }
        Camera *GetCamera() const { return m_camera; }
        RenderGraph *GetRenderGraph() const { return m_renderGraph; }
        Viewport GetViewport() const { return m_viewport; }
        ClearOptions GetClearOptions() const { return m_clearOptions; }

    private:
        Scene *m_scene = nullptr;
        Camera *m_camera = nullptr;
        RenderGraph *m_renderGraph = nullptr;
        Viewport m_viewport;
        ClearOptions m_clearOptions;
    };

} // namespace Engine::Render

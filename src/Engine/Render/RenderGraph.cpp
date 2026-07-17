#include "Engine/Render/RenderGraph.h"

#include <stdexcept>

namespace Engine::Render {

    RenderGraph &RenderGraph::AddPass(std::unique_ptr<RenderPass> pass) {
        if (!pass)
            throw std::runtime_error("RenderGraph::AddPass received null pass");
        m_passes.push_back(std::move(pass));
        return *this;
    }

    void RenderGraph::Clear() {
        m_passes.clear();
    }

    void RenderGraph::Execute(RenderContext &context) const {
        for (const auto &pass: m_passes)
            pass->Execute(context);
    }

} // namespace Engine::Render

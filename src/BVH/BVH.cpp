#include "BVH/BVH.h"

#include <stdexcept>

void BVH::Build(Engine::Core::Context &context, BVHConfiguration config) {
    m_context = &context;
    m_config = config;

    m_backend = MakeBVHBackend(config.backend);
    if (!m_backend)
        throw std::runtime_error("BVH::Build: unknown backend '" + config.backend + "'");
    m_backend->Build(context, config.backendConfig);
}

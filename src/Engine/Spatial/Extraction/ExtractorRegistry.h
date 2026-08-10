#pragma once

#include "Engine/Spatial/Extraction/IsoSurfaceExtractor.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace Engine::Spatial::Extraction {

    class ExtractorRegistry {
    public:
        using Factory = std::function<std::unique_ptr<IsoSurfaceExtractor>()>;

        void Register(const std::string &name, Factory factory) { m_factories[name] = std::move(factory); }
        std::unique_ptr<IsoSurfaceExtractor> Create(const std::string &name) const {
            const auto it = m_factories.find(name);
            return it == m_factories.end() ? nullptr : it->second();
        }
        bool Has(const std::string &name) const { return m_factories.count(name) != 0; }

        static ExtractorRegistry Default();

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace Engine::Spatial::Extraction
